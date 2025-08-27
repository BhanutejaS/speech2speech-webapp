import os
import sys
import queue
import time
import pandas as pd
import torch
import sounddevice as sd
import numpy as np
from datetime import datetime
from transformers import pipeline

# ===== ALLOWED MODELS =====
ALLOWED_MODELS = {
    "tiny": "openai/whisper-tiny.en",
    "base": "openai/whisper-base.en",
    "small": "openai/whisper-small.en",
    "large": "openai/whisper-large-v3-turbo",  # multilingual
    "wav2vec": "facebook/wav2vec2-base-960h"
}

# Pick model from command-line argument
# Example: python realtime_asr.py tiny
choice = sys.argv[1] if len(sys.argv) > 1 else "tiny"

if choice not in ALLOWED_MODELS:
    print(f" Invalid choice '{choice}'. Use one of: {list(ALLOWED_MODELS.keys())}")
    sys.exit(1)

MODEL_ID = ALLOWED_MODELS[choice]

# ===== OUTPUT =====
OUTPUT_DIR = "/Users/bhanu/Desktop/stos/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio params
SAMPLING_RATE = 16000
BLOCK_SIZE = 5  # seconds per chunk

# Setup queue
audio_queue = queue.Queue()

# Torch device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=0 if device.startswith("cuda") else -1,
    torch_dtype=dtype if "whisper" in MODEL_ID else torch.float32  # wav2vec needs float32
)

# Run ID
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

all_results = []

# Callback for sounddevice
def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}", flush=True)
    audio_queue.put(indata.copy())

print(f"Starting real-time speech recognition with {MODEL_ID}. Speak into your mic... (Ctrl+C to stop)")

try:
    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLING_RATE * BLOCK_SIZE),
        callback=callback,
    ):
        while True:
            audio_chunk = audio_queue.get()
            audio_chunk = audio_chunk.flatten()

            start = time.time()
            result = pipe({"array": audio_chunk, "sampling_rate": SAMPLING_RATE})
            end = time.time()

            pred_text = result["text"].strip()
            latency = end - start
            duration = len(audio_chunk) / SAMPLING_RATE
            rtf = latency / duration if duration > 0 else float("inf")

            print(f"\nYou said: {pred_text}")
            print(f"(Latency: {latency:.2f}s, Duration: {duration:.2f}s, RTF={rtf:.2f})")

            all_results.append({
                "run_id": run_id,
                "model": MODEL_ID,
                "duration_sec": duration,
                "latency_sec": latency,
                "RTF": rtf,
                "predicted_text": pred_text
            })

except KeyboardInterrupt:
    print("\n Stopped recording. Saving results...")

    # Save detailed results
    detailed_file = os.path.join(OUTPUT_DIR, "realtime_results.csv")
    pd.DataFrame(all_results).to_csv(detailed_file, index=False)
    print(f"Detailed results saved to {detailed_file}")

    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.groupby(["run_id", "model"]).agg(
            files=("predicted_text", "count"),
            avg_latency_sec=("latency_sec", "mean"),
            avg_RTF=("RTF", "mean"),
            avg_duration_sec=("duration_sec", "mean")
        ).reset_index()

        summary_file = os.path.join(OUTPUT_DIR, "realtime_summary.csv")
        summary.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
