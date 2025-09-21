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

# Set the PHONEMIZER_ESPEAK_LIBRARY environment variable.
# This part is crucial for the kokoro library to find its eSpeak dependency.
# You might need to adjust the path based on your system's eSpeak NG installation.
# Windows:
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
# Linux:
# os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"/usr/lib/libespeak-ng.so"

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
tts_q = queue.Queue()
web_q = queue.Queue()  # New queue for sending messages to the web UI
user_input_q = queue.Queue()  # New queue for receiving text input from the web UI

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
LLM_LOCK = threading.Lock()

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
        item = tts_q.get()
        if item is None:
            break
        sd.play(item, 24000, blocking=True)
        tts_q.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

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
            tts_q.put(chunk)

# ----------------------------- LLM --------------------------------
def speculative_gpt_tts_stream(segment_text):
    global conversation_context
    with LLM_LOCK:
        # Send user message to the web queue
        web_q.put({"role": "user", "content": segment_text})
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT] {segment_text}")

        try:
            buf = ""
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )
            for chunk in resp:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    buf += delta.content
                    print(delta.content, end="", flush=True)
            print("\n[GPT Complete]")
            
            # Send assistant's full response to the web queue
            web_q.put({"role": "assistant", "content": buf})

            speak_by_clauses(buf)
            conversation_context.append({"role": "assistant", "content": buf})

        except Exception as e:
            print(f"[GPT/TTS Error] {e}")

# --------------------------- CORE THREAD ----------------------------
def live_transcribe_loop():
    global last_user_text
    print("Core thread started...")
    model = whisper.load_model(WHISPER_MODEL)
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback
        ):
            while True:
                # Check for input from the web
                if not user_input_q.empty():
                    text = user_input_q.get()
                    if text:
                        threading.Thread(
                            target=speculative_gpt_tts_stream,
                            args=(text,),
                            daemon=True
                        ).start()
                    continue

                # Process audio input
                audio = audio_q.get()
                buffer = np.concatenate((buffer, audio))

                if is_speech(audio):
                    last_speech_time = time.time()

                enough_audio = buffer.size >= int(MIN_CHUNK_SEC * SAMPLE_RATE)
                paused_long = last_speech_time and (time.time() - last_speech_time > PAUSE_TO_CUT_SEC)

                if enough_audio and paused_long:
                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    result = model.transcribe(audio_chunk, **WHISPER_ARGS)
                    text = (result.get("text") or "").strip()

                    if text:
                        sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                        if sim > 0.85 or text.lower() in last_user_text.lower():
                            continue
                        last_user_text = text

                        if not LLM_LOCK.locked():
                            threading.Thread(
                                target=speculative_gpt_tts_stream,
                                args=(text,),
                                daemon=True
                            ).start()
                        
                        rms_values.clear()

    except KeyboardInterrupt:
        print("\nStopped.")
        tts_q.put(None)
    except Exception as e:
        print(f"Core loop error: {e}")

# Start the core logic in a background thread
core_thread = threading.Thread(target=live_transcribe_loop, daemon=True)
core_thread.start()

# --------------------------- FLASK APP ---------------------------
app = Flask(__name__)

# Route to serve the HTML file
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint to get new messages for the web UI
@app.route("/get_updates")
def get_updates():
    messages = []
    while not web_q.empty():
        messages.append(web_q.get())
    return jsonify(messages=messages)

# API endpoint to receive text input from the web UI
@app.route("/send_text", methods=["POST"])
def send_text():
    data = request.json
    text = data.get("text")
    if text:
        user_input_q.put(text)
        return jsonify(success=True)
    return jsonify(success=False, error="No text provided")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
