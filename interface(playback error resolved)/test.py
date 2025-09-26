import os
import queue
import threading
import time
import pathlib
import difflib

import numpy as np
import sounddevice as sd
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from kokoro import KPipeline

from flask import Flask, render_template, jsonify, request

# ----------------------------- CONFIG -----------------------------
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

# VAD / segmentation
RMS_HISTORY = 30
SILENCE_MULTIPLIER = 1.6
MIN_CHUNK_SEC = 0.80
PAUSE_TO_CUT_SEC = 0.90

# Whisper
WHISPER_MODEL = "tiny.en"
WHISPER_ARGS = dict(
    fp16=False,
    language="en",
    condition_on_previous_text=False,
    temperature=0.0,
    no_speech_threshold=0.6,
    logprob_threshold=-0.5,
)

# LLM
LLM_MODEL = "gpt-4o-mini"

# Kokoro TTS
VOICE = "af_heart"
OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="a")

# -------------------------- STATE / QUEUES ------------------------
audio_q = queue.Queue()
tts_q = queue.Queue()
web_q = queue.Queue()
stop_event = threading.Event()
listening_event = threading.Event()  # control input stream

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
LLM_LOCK = threading.Lock()

# -------------------------- LATENCY TRACKER -----------------------
latency_marks = {}

def reset_latency():
    global latency_marks
    latency_marks = {}

# -------------------------- AUDIO HELPERS -------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

def is_speech(audio_chunk):
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def tts_worker():
    while True:
        try:
            if stop_event.is_set():
                print("\n[TTS Stop Signal Received]")
                sd.stop()
                with tts_q.mutex:
                    tts_q.queue.clear()
                stop_event.clear()
                continue

            item = tts_q.get(timeout=0.1)
            sd.play(item, 24000, blocking=False)

            while sd.get_stream().active:
                if stop_event.is_set():
                    break
                time.sleep(0.01)

            if stop_event.is_set():
                sd.stop()
                with tts_q.mutex:
                    tts_q.queue.clear()
                stop_event.clear()

            tts_q.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS worker error: {e}")

threading.Thread(target=tts_worker, daemon=True).start()

# ---------------------------- TTS --------------------------------
def synth_kokoro_stream(text: str, voice: str = VOICE):
    if not text or not text.strip():
        return

    for _gs, _ps, audio in pipe(text.strip(), voice=voice):
        if audio is None or len(audio) == 0:
            continue

        # [LATENCY] First audio emitted (safe + corrected)
        if "tts_first_audio" not in latency_marks:
            latency_marks["tts_first_audio"] = time.time()
            speech_end = latency_marks.get("speech_end")
            llm_first_token = latency_marks.get("llm_first_token")

            if llm_first_token:
                print(f"[Latency] TTS First Audio: {latency_marks['tts_first_audio'] - llm_first_token:.3f}s")
            else:
                print(f"[Latency] TTS First Audio marker set (waiting for llm_first_token)")

            if speech_end:
                print(f"[Latency] End-to-End (STT->TTS): {latency_marks['tts_first_audio'] - speech_end:.3f}s")

        # normalize audio
        audio = np.asarray(audio, dtype=np.float32)
        m = np.max(np.abs(audio))
        if m > 0:
            audio = audio / m
        yield audio



def speak_by_clauses(text: str):
    clauses = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!;:" and len(buf.strip()) > 0:
            clauses.append(buf.strip())
            buf = ""
    if buf.strip():
        clauses.append(buf.strip())

    for clause in clauses:
        for chunk in synth_kokoro_stream(clause):
            if stop_event.is_set():
                return
            tts_q.put(chunk)

# ----------------------------- LLM + TTS -----------------------------
def speculative_gpt_tts_stream(segment_text):
    global conversation_context
    with LLM_LOCK:
        web_q.put({"role": "user", "content": segment_text})
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT] {segment_text}")

        listening_event.clear()  # pause mic

        buf = ""
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )

            for chunk in resp:
                if stop_event.is_set():
                    break
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:

                    # [LATENCY] First GPT token
                    if "llm_first_token" not in latency_marks:
                        latency_marks["llm_first_token"] = time.time()
                        print(f"[Latency] LLM First Token: {latency_marks['llm_first_token'] - latency_marks['speech_end']:.3f}s")

                    buf += delta.content
                    print(delta.content, end="", flush=True)

                    # clause boundary reached → send to TTS immediately
                    if any(c in buf for c in ".?!,") and buf.strip():
                        clause, buf = buf.strip(), ""
                        web_q.put({"role": "assistant", "content": clause})
                        conversation_context.append({"role": "assistant", "content": clause})
                        threading.Thread(
                            target=speak_by_clauses, args=(clause,), daemon=True
                        ).start()

            if buf.strip():
                web_q.put({"role": "assistant", "content": buf.strip()})
                conversation_context.append({"role": "assistant", "content": buf.strip()})
                speak_by_clauses(buf.strip())

            print("\n[GPT Complete]")

        except Exception as e:
            print(f"[GPT/TTS Error] {e}")
        finally:
            tts_q.join()  # wait for TTS playback
            listening_event.set()  # resume mic
            print("[Listening Resumed]")

# --------------------------- CORE THREAD ----------------------------
def live_transcribe_loop():
    global last_user_text
    print("Core thread started...")
    model = whisper.load_model(WHISPER_MODEL)
    buffer = np.zeros((0,), dtype=np.float32)

    try:
        listening_event.set()
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback
        ):
            while True:
                listening_event.wait()
                audio = audio_q.get()
                buffer = np.concatenate((buffer, audio))

                # process every 0.5s instead of waiting for long silence
                if len(buffer) >= int(0.5 * SAMPLE_RATE):
                    chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)

                    # [LATENCY] Mark end of speech chunk
                    latency_marks["speech_end"] = time.time()

                    result = model.transcribe(chunk, **WHISPER_ARGS)
                    text = (result.get("text") or "").strip()

                    # [LATENCY] Whisper complete
                    latency_marks["stt_done"] = time.time()
                    print(f"[Latency] STT: {latency_marks['stt_done'] - latency_marks['speech_end']:.3f}s")

                    if text:
                        sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                        if sim > 0.85 or text.lower() in last_user_text.lower():
                            continue
                        last_user_text = text

                        stop_event.clear()
                        if not LLM_LOCK.locked():
                            reset_latency()
                            latency_marks["speech_end"] = time.time()
                            threading.Thread(
                                target=speculative_gpt_tts_stream,
                                args=(text,),
                                daemon=True
                            ).start()
                        rms_values.clear()
    except Exception as e:
        print(f"Core loop error: {e}")


core_thread = threading.Thread(target=live_transcribe_loop, daemon=True)
core_thread.start()

# --------------------------- FLASK APP ---------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_updates")
def get_updates():
    messages = []
    while not web_q.empty():
        messages.append(web_q.get())
    return jsonify(messages=messages)

@app.route("/stop_speaking")
def stop_speaking():
    print("Received Stop Signal from Web UI")
    stop_event.set()
    listening_event.set()  # resume listening immediately
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
