Real-Time Speech-to-Speech AI Assistant

This project is a real-time speech-to-speech system that captures user audio, transcribes it using OpenAI Whisper, sends the transcription to GPT for conversational responses, and returns the results in real time.

Key Features:

Live speech recognition with adaptive pause detection (Whisper).

Low-latency GPT responses streamed back as soon as the model starts replying.

Latency benchmarking – measures and logs the delay between end of speech and GPT’s first response (saved in CSV).

Scalable architecture ready for speech-to-speech integration (text-to-speech can be added).

Web app ready – backend logic can be integrated into a browser interface for cross-platform access.

Tech Stack:

Python for audio capture, transcription, and response handling.

OpenAI Whisper for speech-to-text.

GPT (OpenAI API) for natural language conversation.

SoundDevice + NumPy for real-time audio streaming.

CSV logging for benchmarking and performance evaluation.

Future: Web frontend streamlit for browser-based interaction.

 Use Cases:

AI voice assistants

Interactive learning tutors

Accessibility tools (hands-free interaction)

Real-time meeting or interview support
