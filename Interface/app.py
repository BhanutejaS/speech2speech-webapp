import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

import queue
import threading
import time
import pathlib
import difflib
import base64
import numpy as np
import sounddevice as sd
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from kokoro import KPipeline
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

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
llm_q = queue.Queue() # Queue to handle incoming LLM requests

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
whisper_model = whisper.load_model(WHISPER_MODEL)
LLM_LOCK = threading.Lock()

# -------------------------- FLASK APP SETUP -------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------- AUDIO PROCESSING ----------------------------
def is_speech(audio_chunk):
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

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
            # Send TTS audio back over WebSocket
            socketio.emit('audio_chunk', chunk.tobytes())

# ----------------------------- LLM --------------------------------
def speculative_gpt_tts_stream(segment_text):
    with LLM_LOCK:
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT] {segment_text}")
        socketio.emit('text_update', f"[User]: {segment_text}")

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
                    # Emit incremental text to the frontend
                    socketio.emit('text_update', delta.content)
            print("\n[GPT Complete]")
            speak_by_clauses(buf)
            conversation_context.append({"role": "assistant", "content": buf})

        except Exception as e:
            print(f"[GPT/TTS Error] {e}")

# --------------------------- SOCKETIO HANDLERS ----------------------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    # Decode audio data from base64 and convert to numpy array
    audio_data = np.frombuffer(base64.b64decode(data), dtype=np.float32)
    audio_q.put(audio_data)

# --------------------------- MAIN LOOP ----------------------------
def process_audio_chunks():
    global last_user_text
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None
    while True:
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

            result = whisper_model.transcribe(audio_chunk, **WHISPER_ARGS)
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
                else:
                    pass

                rms_values.clear()

# Start the audio processing thread
threading.Thread(target=process_audio_chunks, daemon=True).start()

if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)