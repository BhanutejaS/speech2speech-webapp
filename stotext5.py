import whisper
import sounddevice as sd
import numpy as np
import os
import queue
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------- CONFIG -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper model (tiny English for low latency)
model = whisper.load_model("tiny.en")

# Audio parameters
samplerate = 16000
blocksize = 512
q = queue.Queue()

# Adaptive RMS pause detection parameters
RMS_HISTORY = 30          # Tracks recent RMS to adapt to background noise
SILENCE_MULTIPLIER = 1.5 # Speech threshold multiplier
MIN_CHUNK_SIZE = samplerate // 2  # Minimum 0.5 sec audio

rms_values = []

# -------------------------- FUNCTIONS ----------------------------

def audio_callback(indata, frames, time_info, status):
    """Callback to put microphone audio into the queue."""
    if status:
        print(status)
    q.put(indata.copy())

def is_speech(audio_chunk):
    """Check if audio chunk is speech using adaptive RMS threshold."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values)
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def stream_gpt_response_partial(segment_text, segment_index, speech_end_time):
    """
    Stream GPT responses using speculative sampling.
    Starts printing responses as soon as they arrive instead of waiting for full generation.
    """
    print(f"\n[STT -> GPT] Segment {segment_index}: {segment_text}")
    try:
        # Streaming completion request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": segment_text}
            ],
            stream=True  # Enables input streaming
        )

        # Track first output chunk to measure latency
        first_chunk = True
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                if first_chunk:
                    gpt_start_time = time.time()
                    latency = gpt_start_time - speech_end_time
                    print(f"[Latency] Segment {segment_index}: {latency:.3f} sec")
                    first_chunk = False
                print(delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"[GPT Error] {e}")

# --------------------------- MAIN LOOP ---------------------------

def live_transcribe_and_stream_chat():
    print("Start speaking (Ctrl+C to stop)...")
    buffer = np.zeros((0,), dtype=np.float32)
    segment_index = 1
    last_speech_time = None

    try:
        with sd.InputStream(
            samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback
        ):
            while True:
                chunk = q.get()
                audio_data = chunk[:, 0]
                buffer = np.concatenate((buffer, audio_data))

                # Detect speech
                if is_speech(audio_data):
                    last_speech_time = time.time()

                # Trigger transcription after minimum chunk and short human-like pause
                if buffer.size >= MIN_CHUNK_SIZE and last_speech_time and (time.time() - last_speech_time > 0.3):
                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    # Transcribe with Whisper
                    result = model.transcribe(audio_chunk, fp16=False)
                    text = result["text"].strip()

                    if text:
                        # Use thread to asynchronously stream GPT response
                        threading.Thread(
                            target=stream_gpt_response_partial,
                            args=(text, segment_index, time.time()),
                            daemon=True
                        ).start()
                        segment_index += 1

    except KeyboardInterrupt:
        print("\nStopped live transcription.")

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    live_transcribe_and_stream_chat()
