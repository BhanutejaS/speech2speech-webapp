import whisper
import sounddevice as sd
import numpy as np
import queue
import os
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv
import pathlib
from kokoro import KPipeline

# ----------------------------- CONFIG -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper STT
model = whisper.load_model("tiny.en")
samplerate = 16000
blocksize = 512
q = queue.Queue()
RMS_HISTORY = 30
SILENCE_MULTIPLIER = 1.5
MIN_CHUNK_SIZE = samplerate // 2  # 0.5 sec
rms_values = []

# Conversation context
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]

# Kokoro TTS
VOICE = "af_heart"
OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="a")

# -------------------------- AUDIO QUEUE --------------------------
tts_queue = queue.Queue()

def tts_worker():
    """Continuously play audio chunks from the queue (blocking)."""
    while True:
        audio_chunk = tts_queue.get()
        if audio_chunk is None:
            break
        sd.play(audio_chunk, 24000, blocking=True)
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# -------------------------- FUNCTIONS ----------------------------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def is_speech(audio_chunk):
    """Detect speech based on RMS with adaptive threshold."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values)
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def synth_kokoro_stream(text: str, voice: str = VOICE):
    """Generator: yield clean audio chunks as they are produced."""
    if not text or not text.strip():
        return
    for _gs, _ps, audio in pipe(text.strip(), voice=voice):
        if audio is None or len(audio) == 0:
            continue
        audio = np.asarray(audio, dtype=np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        yield audio

def speculative_gpt_tts_stream(segment_text):
    """Stream GPT output and generate TTS word-by-word."""
    conversation_context.append({"role": "user", "content": segment_text})
    print(f"\n[STT -> GPT] {segment_text}")

    try:
        buffer_for_tts = ""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_context,
            stream=True
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                buffer_for_tts += delta.content

                # Split by words and generate TTS for every 4-word segment
                words = buffer_for_tts.split()
                while len(words) >= 4:
                    to_speak = " ".join(words[:4])
                    words = words[4:]
                    for audio_chunk in synth_kokoro_stream(to_speak, VOICE):
                        tts_queue.put(audio_chunk)
                    buffer_for_tts = " ".join(words)

                print(delta.content, end="", flush=True)

        # Final TTS for remaining words
        if buffer_for_tts.strip():
            for audio_chunk in synth_kokoro_stream(buffer_for_tts, VOICE):
                tts_queue.put(audio_chunk)

        conversation_context.append({"role": "assistant", "content": buffer_for_tts})
        print("\n[GPT Complete]")

    except Exception as e:
        print(f"[GPT/TTS Error] {e}")

# --------------------------- MAIN LOOP ---------------------------

def live_transcribe_gpt_tts_stream():
    print("Start speaking (Ctrl+C to stop)...")
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(
            samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback
        ):
            while True:
                chunk = q.get()
                audio_data = chunk[:, 0]
                buffer = np.concatenate((buffer, audio_data))

                if is_speech(audio_data):
                    last_speech_time = time.time()

                # Trigger after short pause (0.3s)
                if buffer.size >= MIN_CHUNK_SIZE and last_speech_time and (time.time() - last_speech_time > 0.3):
                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    result = model.transcribe(audio_chunk, fp16=False)
                    text = result["text"].strip()

                    if text:
                        threading.Thread(
                            target=speculative_gpt_tts_stream,
                            args=(text,),
                            daemon=True
                        ).start()

    except KeyboardInterrupt:
        print("\nStopped live transcription.")
        tts_queue.put(None)  # stop TTS worker

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    live_transcribe_gpt_tts_stream()
