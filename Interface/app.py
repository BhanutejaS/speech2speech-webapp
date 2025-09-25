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
web_q = queue.Queue()         # messages to the web UI
user_input_q = queue.Queue()  # text input from the web UI

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
LLM_LOCK = threading.Lock()

STOP_SIGNAL = threading.Event()   # global stop/cancel flag
TTS_PLAYING = False
STT_PAUSED = False
VOLUME = 0.7
AUDIO_DEVICE = None
CURRENT_AUDIO_LEVEL = 0.0

# NEW: monotonically increasing id to cancel in-flight work
RESPONSE_GEN_ID = 0

# -------------------------- AUDIO HELPERS -------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

def is_speech(audio_chunk):
    global CURRENT_AUDIO_LEVEL
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))
    CURRENT_AUDIO_LEVEL = min(1.0, rms * 10)  # Scale for visualizer
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def tts_worker():
    """Continuously plays audio chunks from tts_q. Respects STOP_SIGNAL."""
    global TTS_PLAYING
    while True:
        item = tts_q.get()
        if item is None:
            # shutdown signal for worker
            tts_q.task_done()
            break

        # If we were asked to stop, discard this chunk and continue
        if STOP_SIGNAL.is_set():
            tts_q.task_done()
            continue

        # Apply volume scaling
        scaled_audio = item * VOLUME
        device_id = AUDIO_DEVICE if AUDIO_DEVICE != "default" else None

        # Mark TTS as playing to mute microphone
        TTS_PLAYING = True

        # Start non-blocking playback
        try:
            sd.play(scaled_audio, 24000, device=device_id, blocking=False)
        except Exception as e:
            print(f"[TTS play error] {e}")
            TTS_PLAYING = False
            tts_q.task_done()
            continue

        # While audio is playing, allow Stop to cut it off
        while True:
            try:
                s = sd.get_stream()
                active = bool(s and getattr(s, "active", False))
            except Exception:
                # Fallback if stream lookup fails
                active = False

            if not active:
                break

            if STOP_SIGNAL.is_set():
                try:
                    sd.stop()
                except Exception:
                    pass
                break

            time.sleep(0.01)

        # Mark TTS as stopped
        TTS_PLAYING = False
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
    """Chunk TTS by simple clause boundaries and respect STOP_SIGNAL."""
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
        if STOP_SIGNAL.is_set():
            return  # Exit immediately if stopped
        for chunk in synth_kokoro_stream(clause, VOICE):
            if STOP_SIGNAL.is_set():
                return  # Exit immediately if stopped
            tts_q.put(chunk)

# ----------------------------- LLM --------------------------------
def speculative_gpt_tts_stream(segment_text):
    """
    Streams LLM tokens to the UI and (if not cancelled) to TTS.
    Cancellation is done via STOP_SIGNAL or RESPONSE_GEN_ID changes.
    """
    global conversation_context

    with LLM_LOCK:
        # Start of a *new* response: allow talking again
        STOP_SIGNAL.clear()
        my_gen_id = RESPONSE_GEN_ID

        # Send user message to UI and context
        web_q.put({"role": "user", "content": segment_text})
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT] {segment_text}")

        try:
            buf = ""
            message_id = f"msg_{int(time.time() * 1000)}"  # Unique ID for this message

            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )

            cancelled = False
            for chunk in resp:
                # CANCEL CHECK
                if STOP_SIGNAL.is_set() or my_gen_id != RESPONSE_GEN_ID:
                    cancelled = True
                    break

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    buf += delta.content
                    print(delta.content, end="", flush=True)
                    # Send streaming update to web queue
                    web_q.put({
                        "role": "assistant",
                        "content": buf,
                        "streaming": True,
                        "message_id": message_id
                    })

            # Close the underlying stream if possible
            try:
                resp.close()
            except Exception:
                pass

            print("\n[GPT Cancelled]" if cancelled else "\n[GPT Complete]")

            if cancelled:
                # Finalize streaming message as "(stopped)" and do not speak or append to history
                web_q.put({
                    "role": "assistant",
                    "content": "(stopped)",
                    "streaming": False,
                    "message_id": message_id
                })
                return

            # Finalize message
            web_q.put({
                "role": "assistant",
                "content": buf,
                "streaming": False,
                "message_id": message_id
            })

            # Speak only if not stopped and still current gen
            if not STOP_SIGNAL.is_set() and my_gen_id == RESPONSE_GEN_ID:
                speak_by_clauses(buf)
                # Only append to history if fully delivered
                conversation_context.append({"role": "assistant", "content": buf})

        except Exception as e:
            print(f"[GPT/TTS Error] {e}")

# --------------------------- CORE THREAD ----------------------------
def live_transcribe_loop():
    """Captures mic, segments with simple VAD, sends text to LLM."""
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
                # Check for typed input from the web UI
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

                # Skip audio processing if STT is paused or TTS is playing
                if STT_PAUSED or TTS_PLAYING:
                    continue

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

# API endpoint to stop AI response
@app.route("/stop", methods=["POST"])
def stop_response():
    global TTS_PLAYING, RESPONSE_GEN_ID
    # Signal stop and invalidate in-flight work
    STOP_SIGNAL.set()
    RESPONSE_GEN_ID += 1

    # Stop any currently playing audio and flush device buffers
    try:
        sd.stop()
    except Exception:
        pass

    TTS_PLAYING = False  # Unmute microphone when stopping

    # Clear TTS queue completely
    while not tts_q.empty():
        try:
            tts_q.get_nowait()
            tts_q.task_done()
        except queue.Empty:
            break

    # IMPORTANT: do NOT auto-clear STOP_SIGNAL here.
    # It will be cleared at the start of the next request in speculative_gpt_tts_stream.
    return jsonify(success=True)

# API endpoint to toggle STT pause
@app.route("/toggle_stt", methods=["POST"])
def toggle_stt():
    global STT_PAUSED
    data = request.json
    STT_PAUSED = data.get("paused", False)
    return jsonify(success=True, paused=STT_PAUSED)

# API endpoint to change TTS voice
@app.route("/change_voice", methods=["POST"])
def change_voice():
    global VOICE
    data = request.json
    VOICE = data.get("voice", VOICE)
    return jsonify(success=True, voice=VOICE)

# API endpoint to change silence multiplier
@app.route("/change_silence_multiplier", methods=["POST"])
def change_silence_multiplier():
    global SILENCE_MULTIPLIER
    data = request.json
    SILENCE_MULTIPLIER = data.get("multiplier", SILENCE_MULTIPLIER)
    return jsonify(success=True, multiplier=SILENCE_MULTIPLIER)

# API endpoint to change volume
@app.route("/change_volume", methods=["POST"])
def change_volume():
    global VOLUME
    data = request.json
    VOLUME = data.get("volume", VOLUME)
    return jsonify(success=True, volume=VOLUME)

# API endpoint to change audio device
@app.route("/change_audio_device", methods=["POST"])
def change_audio_device():
    global AUDIO_DEVICE
    data = request.json
    AUDIO_DEVICE = data.get("device_id", "default")
    return jsonify(success=True, device_id=AUDIO_DEVICE)

# API endpoint to get available audio devices
@app.route("/get_audio_devices")
def get_audio_devices():
    try:
        devices = sd.query_devices()
        output_devices = []
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'id': i,
                    'name': device['name']
                })
        return jsonify(devices=output_devices)
    except Exception as e:
        return jsonify(devices=[], error=str(e))

# API endpoint to get current audio level
@app.route("/get_audio_level")
def get_audio_level():
    return jsonify(level=CURRENT_AUDIO_LEVEL)

if __name__ == "__main__":
    # Ensure templates folder exists for index.html
    # Run: python app.py, then open http://127.0.0.1:5000/
    app.run(host="127.0.0.1", port=5000, debug=False)
