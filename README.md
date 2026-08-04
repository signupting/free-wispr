# Free Wispr

A fast, local-first dictation app that lives in the macOS menu bar. Tap a key,
talk, and your speech is transcribed and pasted wherever your cursor is — no
subscription, no per-minute API bill, no audio leaving your machine.

## How it works

- **Trigger:** single-tap **Right Option (⌥)** to start/stop recording;
  double-tap to open the clipboard/history picker.
- **Transcription:** on-device via **NVIDIA Parakeet TDT 0.6B** (Apple MLX).
  It's fast (a 30–60s dictation transcribes in ~2s), has punctuation and
  capitalization built in, and works fully offline.
- **Cloud fallbacks:** if the local model ever fails, it falls back to
  **Groq Whisper** and then **HuggingFace** — so a dictation is never lost.
- **Clean output:** leading/trailing silence is trimmed and filler words
  ("um", "uh") are stripped so the pasted text reads cleanly.

## Requirements

- Apple Silicon Mac (built/tested on an M1).
- Python with the app's deps: `mlx`, `parakeet-mlx`, `sounddevice`, `scipy`,
  `numpy`, `pyobjc`, `groq`, plus `requests` for the HF fallback.

## Setup

1. Create a `.env` in this folder for the cloud fallbacks (optional but
   recommended):

   ```
   GROQ_API_KEY=your_groq_key
   HF_API_KEY=your_hf_key
   ```

2. Build the quantized local model once (8-bit — smaller, so it reloads fast on
   RAM-tight machines):

   ```sh
   ~/anaconda3/python.app/Contents/MacOS/python tools/build_quantized.py
   ```

   This writes `parakeet-8bit/` (~734 MB). If it's missing, the app falls back to
   downloading the full-precision Parakeet model from HuggingFace on first run.

3. Launch:

   ```sh
   ./run-free-wispr.sh
   ```

## Notes

- **Local-first:** Parakeet runs on-device; Groq/HF are only touched if the
  local path fails. Set `.env` keys if you want the safety net.
- **Privacy:** recorded audio and transcription history stay on your machine
  (`backups/`, `history.json`) and are gitignored.
- **8 GB Macs:** the app keeps the model warm in the background so it isn't
  evicted from RAM between dictations. If dictation still stalls after long idle
  gaps, the machine is under memory pressure — freeing RAM helps.
