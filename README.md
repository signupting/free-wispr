# Free Wispr

Free, open-source macOS speech-to-text. Tap **fn** to dictate anywhere.

## Features

- **Tap fn** to start/stop recording — text is pasted at your cursor
- **Groq Whisper API** for fast, accurate transcription (free tier available)
- **HuggingFace fallback** if Groq is down (optional)
- **AI cleanup** via Llama 3.3 — removes filler words, fixes phrasing
- **Menubar icon** shows recording (red) / processing (orange) / idle (gray) state
- **Audio backups** saved automatically, deleted after successful transcription

## Requirements

- macOS (tested on Sonoma/Sequoia)
- Python 3
- [Groq API key](https://console.groq.com/keys) (free)

## Install

```bash
git clone https://github.com/signupting/free-wispr.git
cd free-wispr
bash install.sh
```

The installer will:
1. Install Python dependencies
2. Prompt for your API key(s)
3. Create a macOS app bundle at `/Applications/Free Wispr.app`
4. Launch the app

## Permissions

After first launch, grant these in **System Settings > Privacy & Security**:

1. **Microphone** — required for recording
2. **Accessibility** — required for fn key detection and text pasting

You may need to restart the app after granting permissions.

## Usage

1. Open any text field (editor, chat, browser, etc.)
2. **Tap fn** once to start recording (icon turns red)
3. **Tap fn** again to stop (icon turns orange while processing)
4. Transcribed text is automatically pasted at your cursor

## How it works

- A persistent audio stream listens via CoreAudio
- fn key is detected via NSTimer polling + NSEvent monitors
- Audio is sent to Groq's Whisper API (falls back to HuggingFace)
- Llama 3.3 cleans up the raw transcription
- Result is copied to clipboard and pasted via `Cmd+V`

## Troubleshooting

- **No response to fn key**: Check Accessibility permission. Restart the app.
- **"Too quiet" in logs**: Speak closer to the mic. Check `tail -f /tmp/groq-whisper.log`
- **API errors**: Verify your key at https://console.groq.com
- **App not starting**: Run manually: `~/.local/groq-whisper-app/groq_whisper.py`

## Uninstall

```bash
bash uninstall.sh
```
