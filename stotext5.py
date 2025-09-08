import whisper
import sounddevice as sd
import numpy as np
import queue
from openai import OpenAI
import threading
import time
import csv
import os
from dotenv import load_dotenv

# ----------------------------- CONFIG -----------------------------
# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Folder to save CSV
output_folder = "/Users/bhanu/Desktop/benchmarks"
os.makedirs(output_folder, exist_ok=True)
csv_file_path = os.path.join(output_folder, "speech_to_agent_latency.csv")

# Whisper model
model = whisper.load_model("tiny.en")

# Audio parameters
samplerate = 16000
blocksize = 512
q = queue.Queue()

# Pause detection params
PAUSE_THRESHOLD = 1.0  # seconds of silence to consider speech ended
RMS_HISTORY = 20
silence_multiplier = 0.3
rms_values = []

# CSV header
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["SegmentIndex", "STT_Text", "SpeechEndTime", "GPTStartTime", "LatencySeconds"]
        )

# -------------------------- FUNCTIONS ----------------------------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def compute_rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk**2))

def stream_gpt_response(segment_text, segment_index, speech_end_time):
    """Stream GPT response and measure latency to start of response"""
    print("\n[GPT] ", end="", flush=True)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": segment_text}
            ],
            stream=True
        )

        first_chunk = True
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                if first_chunk:
                    gpt_start_time = time.time()
                    latency = gpt_start_time - speech_end_time
                    # Save latency to CSV
                    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [segment_index, segment_text, speech_end_time, gpt_start_time, latency]
                        )
                    # Print latency in real time
                    print(f"\n[Latency] Segment {segment_index}: {latency:.3f} seconds")
                    first_chunk = False
                print(delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"\n[GPT Error] {e}")

# --------------------------- MAIN LOOP ---------------------------

def live_transcribe_and_stream_chat():
    print("Start speaking (press Ctrl+C to stop)...")
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = time.time()
    segment_size = samplerate // 4  # 0.25s chunk processing
    segment_index = 1

    try:
        with sd.InputStream(
            samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback
        ):
            while True:
                chunk = q.get()
                audio_data = chunk[:, 0]
                buffer = np.concatenate((buffer, audio_data))

                # Adaptive RMS for pause detection
                rms = compute_rms(audio_data)
                rms_values.append(rms)
                if len(rms_values) > RMS_HISTORY:
                    rms_values.pop(0)
                avg_rms = np.mean(rms_values)
                silence_threshold = avg_rms * silence_multiplier

                is_speech = rms > silence_threshold
                if is_speech:
                    last_speech_time = time.time()

                # Pause detected, send to GPT
                if len(buffer) >= segment_size and (time.time() - last_speech_time > PAUSE_THRESHOLD):
                    audio_float = buffer
                    buffer = np.zeros((0,), dtype=np.float32)

                    result = model.transcribe(audio_float, fp16=False)
                    text = result["text"].strip()

                    if text:
                        print(f"\n[STT] {text}")
                        speech_end_time = time.time()
                        threading.Thread(
                            target=stream_gpt_response,
                            args=(text, segment_index, speech_end_time),
                            daemon=True
                        ).start()
                        segment_index += 1

    except KeyboardInterrupt:
        print("\nStopped live transcription.")

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    live_transcribe_and_stream_chat()
