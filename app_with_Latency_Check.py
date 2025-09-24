import os
import queue
import threading
import time
import pathlib
import difflib
from statistics import mean

import numpy as np
import sounddevice as sd
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from kokoro import KPipeline

from flask import Flask, render_template, jsonify, request

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# ----------------------------- CONFIG -----------------------------
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
tts_q = queue.Queue()    # will carry dicts: {"turn_id": int, "audio": np.ndarray}
web_q = queue.Queue()
stop_event = threading.Event()
listening_event = threading.Event()  # controls input stream

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
LLM_LOCK = threading.Lock()

# --------------------- LATENCY / BENCHMARK STATE ------------------
TURN_COUNTER = 0
LAT = {}        # turn_id -> dict of timings & metadata
LAT_LOCK = threading.Lock()
# keys we record per turn:
# "e2e_start", "stt_time", "llm_ttft", "llm_total",
# "tts_ttft_play", "e2e_ttft", "user_text", "assistant_text"

def _new_turn():
    global TURN_COUNTER
    with LAT_LOCK:
        TURN_COUNTER += 1
        turn_id = TURN_COUNTER
        LAT[turn_id] = {}
    return turn_id

def _lat_set(turn_id, key, value):
    with LAT_LOCK:
        if turn_id in LAT:
            LAT[turn_id][key] = value

def _lat_get(turn_id, key, default=None):
    with LAT_LOCK:
        return LAT.get(turn_id, {}).get(key, default)

def _lat_summarize():
    # returns list of per-turn dicts + simple averages
    with LAT_LOCK:
        rows = []
        for tid, d in sorted(LAT.items()):
            row = {"turn_id": tid}
            row.update({
                "stt_time": d.get("stt_time"),
                "llm_ttft": d.get("llm_ttft"),
                "llm_total": d.get("llm_total"),
                "tts_ttft_play": d.get("tts_ttft_play"),
                "e2e_ttft": d.get("e2e_ttft"),
                "user_text": d.get("user_text"),
                "assistant_len": len(d.get("assistant_text", "")) if d.get("assistant_text") else 0,
            })
            rows.append(row)

        def _avg(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return round(mean(vals), 4) if vals else None

        summary = {
            "avg_stt_time": _avg("stt_time"),
            "avg_llm_ttft": _avg("llm_ttft"),
            "avg_llm_total": _avg("llm_total"),
            "avg_tts_ttft_play": _avg("tts_ttft_play"),
            "avg_e2e_ttft": _avg("e2e_ttft"),
            "turns_counted": len(rows),
        }
        return rows, summary

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

# ---------------------------- TTS --------------------------------
def synth_kokoro_stream(text: str, voice: str = VOICE):
    if not text or not text.strip():
        return
    for _gs, _ps, audio in pipe(text.strip(), voice=voice):
        if audio is None or len(audio) == 0:
            continue
        audio = np.asarray(audio, dtype=np.float32)
        m = np.max(np.abs(audio))
        if m > 0:
            audio = audio / m
        yield audio

def speak_by_clauses(text: str, turn_id: int):
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
            # enqueue chunk annotated with turn_id (so tts_worker can log TTFT play)
            tts_q.put({"turn_id": turn_id, "audio": chunk})

# ------------------------ TTS WORKER (PLAY) -----------------------
def tts_worker():
    print("[TTS worker] started")
    # track first-play per turn
    seen_first_play = set()

    while True:
        try:
            if stop_event.is_set():
                print("\n[TTS Stop Signal Received]")
                sd.stop()
                with tts_q.mutex:
                    tts_q.queue.clear()
                stop_event.clear()
                continue

            item = tts_q.get(timeout=0.1)  # item is {"turn_id": int, "audio": np.ndarray}
            if not isinstance(item, dict) or "audio" not in item:
                tts_q.task_done()
                continue

            tid = item.get("turn_id")
            audio = item["audio"]

            # If it's the first chunk that is about to play for this turn, capture TTS play TTFT and E2E TTFT
            if tid and tid not in seen_first_play:
                seen_first_play.add(tid)
                now = time.time()
                _lat_set(tid, "tts_ttft_play", round(now - _lat_get(tid, "llm_start", now), 4))
                e2e_start = _lat_get(tid, "e2e_start")
                if e2e_start:
                    _lat_set(tid, "e2e_ttft", round(now - e2e_start, 4))
                print(f"[Latency][Turn {tid}] TTS TTFT(play): {_lat_get(tid,'tts_ttft_play')} s | "
                      f"E2E TTFT: {_lat_get(tid,'e2e_ttft')} s")

            # Play audio non-blocking and wait for completion (so stop works mid-utterance)
            sd.play(audio, 24000, blocking=False)
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

# ----------------------------- LLM --------------------------------
def speculative_gpt_tts_stream(segment_text, turn_id: int):
    global conversation_context
    with LLM_LOCK:
        web_q.put({"role": "user", "content": segment_text})
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT][Turn {turn_id}] {segment_text}")

        # Pause listening while AI generates/speaks
        listening_event.clear()

        try:
            buf = ""
            llm_start = time.time()
            _lat_set(turn_id, "llm_start", llm_start)

            first_token_time = None
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )
            for chunk in resp:
                if stop_event.is_set():
                    print("\n[LLM Generation Stopped]")
                    break
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        _lat_set(turn_id, "llm_ttft", round(first_token_time - llm_start, 4))
                        print(f"[Latency][Turn {turn_id}] LLM TTFT: {_lat_get(turn_id,'llm_ttft')} s")
                    buf += delta.content
                    print(delta.content, end="", flush=True)

            llm_end = time.time()
            _lat_set(turn_id, "llm_total", round(llm_end - llm_start, 4))
            print(f"\n[GPT Complete][Turn {turn_id}] LLM total: {_lat_get(turn_id,'llm_total')} s")

            web_q.put({"role": "assistant", "content": buf})
            _lat_set(turn_id, "assistant_text", buf)

            if not stop_event.is_set():
                # enqueue to TTS; tts_worker will log TTFT(play)
                speak_by_clauses(buf, turn_id)
                conversation_context.append({"role": "assistant", "content": buf})

        except Exception as e:
            print(f"[GPT/TTS Error][Turn {turn_id}] {e}")
        finally:
            # Wait for all TTS audio to finish playing before resuming listening
            tts_q.join()
            listening_event.set()
            print(f"[Listening Resumed][Turn {turn_id}]")

# --------------------------- CORE THREAD ----------------------------
def live_transcribe_loop():
    global last_user_text
    print("Core thread started...")
    model = whisper.load_model(WHISPER_MODEL)
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        # Start in a listening state
        listening_event.set()
        while True:
            listening_event.wait()
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback
            ):
                while listening_event.is_set():
                    audio = audio_q.get()
                    buffer = np.concatenate((buffer, audio))

                    if is_speech(audio):
                        last_speech_time = time.time()

                    enough_audio = buffer.size >= int(MIN_CHUNK_SEC * SAMPLE_RATE)
                    paused_long = last_speech_time and (time.time() - last_speech_time > PAUSE_TO_CUT_SEC)

                    if enough_audio and paused_long:
                        # Speech ended; mark E2E start for a new turn
                        turn_id = _new_turn()
                        e2e_start = time.time()
                        _lat_set(turn_id, "e2e_start", e2e_start)

                        audio_chunk = buffer.copy()
                        buffer = np.zeros((0,), dtype=np.float32)
                        last_speech_time = None

                        # STT timing
                        stt_t0 = time.time()
                        result = model.transcribe(audio_chunk, **WHISPER_ARGS)
                        stt_t1 = time.time()
                        stt_time = round(stt_t1 - stt_t0, 4)
                        _lat_set(turn_id, "stt_time", stt_time)

                        text = (result.get("text") or "").strip()
                        _lat_set(turn_id, "user_text", text)
                        print(f"[Latency][Turn {turn_id}] STT: {stt_time} s | Text: {text!r}")

                        if text:
                            sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                            if sim > 0.85 or text.lower() in last_user_text.lower():
                                continue
                            last_user_text = text

                            stop_event.clear()

                            if not LLM_LOCK.locked():
                                threading.Thread(
                                    target=speculative_gpt_tts_stream,
                                    args=(text, turn_id),
                                    daemon=True
                                ).start()

                            rms_values.clear()

    except KeyboardInterrupt:
        print("\nStopped.")
        tts_q.put(None)
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
    """Endpoint to stop the TTS playback."""
    print("Received Stop Signal from Web UI")
    stop_event.set()
    listening_event.set()  # Resume listening immediately upon stop
    return jsonify(success=True)

@app.route("/latency_stats")
def latency_stats():
    """
    Returns per-turn latency rows + simple averages.
    Each row includes: turn_id, stt_time, llm_ttft, llm_total, tts_ttft_play, e2e_ttft, user_text, assistant_len.
    """
    rows, summary = _lat_summarize()
    return jsonify(summary=summary, rows=rows)

if __name__ == "__main__":
    # Tip: run with `set FLASK_ENV=development` (Windows) or `export FLASK_ENV=development` (Unix)
    app.run(host="127.0.0.1", port=5000)
