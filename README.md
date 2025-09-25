# Latency Check & Benchmark
This module measures and reports latency benchmarks for the conversational AI pipeline.
It records per-turn timings for speech-to-text (STT), LLM response, text-to-speech (TTS), and end-to-end (E2E) performance.

# Metrics Captured
For every conversation turn, the following are logged:

## Metric	Description

- turn_id	: Conversation turn number
- stt_time :	Time (in seconds) Whisper needed to transcribe speech → text
- llm_ttft :	GPT time-to-first-token (LLM time-to-first-token)
- llm_total	: Total GPT response generation time
- tts_ttft_play	: Time from GPT finishing → first TTS audio chunk played (TTS time-to-first-token play)
- e2e_ttft :	End-to-end time (speech end → first AI audio)
- user_text_len :	Number of characters in transcribed user input
- assistant_len :	Number of characters in GPT’s reply
