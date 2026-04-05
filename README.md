# Free Wispr

Free, open-source speech-to-text for **macOS and Windows**. Tap a hotkey to dictate anywhere.

## Features

- **Hotkey to start/stop** recording — text is pasted at your cursor
- **Groq Whisper API** for fast, accurate transcription (free tier available)
- **HuggingFace fallback** if Groq is down (optional)
- **AI cleanup** via Llama 3.3 — removes filler words, fixes phrasing
- **Tray/menubar icon** shows recording (red) / processing (orange) / idle (gray) state
- **Audio backups** saved automatically, deleted after successful transcription

## Requirements

- Python 3
- [Groq API key](https://console.groq.com/keys) (free)

---

## macOS Install

```bash
git clone https://github.com/signupting/free-wispr.git
cd free-wispr
bash install.sh
```

**Hotkey:** Tap **fn**

**Permissions required** (System Settings → Privacy & Security):
1. **Microphone** — required for recording
2. **Accessibility** — required for fn key detection and text pasting

**Uninstall:** `bash uninstall.sh`

---

## Windows Install

1. Install [Python 3](https://www.python.org/downloads/) — check **"Add to PATH"** during install
2. Clone or download this repo
3. Right-click `install_windows.ps1` → **Run with PowerShell**
   *(or: `powershell -ExecutionPolicy Bypass -File install_windows.ps1`)*

**Hotkey:** `Ctrl+Shift+Space`

The installer will:
1. Install Python dependencies
2. Prompt for your API key(s)
3. Add Free Wispr to Windows startup
4. Launch the app in your system tray

**Custom hotkey:** Set the `FREE_WISPR_HOTKEY` environment variable before running
(e.g. `ctrl+alt+space`, `f9`, `right ctrl`)

**Logs:** `%USERPROFILE%\groq-whisper.log`

---

## Usage

1. Open any text field (editor, chat, browser, etc.)
2. Press the hotkey to **start** recording (icon turns red)
3. Press again to **stop** (icon turns orange while processing)
4. Transcribed text is automatically pasted at your cursor

## How it works

- A persistent audio stream listens in the background
- Hotkey is detected globally (fn polling on macOS, `keyboard` library on Windows)
- Audio is sent to Groq's Whisper API (falls back to HuggingFace)
- Llama 3.3 lightly cleans up the raw transcription
- Result is copied to clipboard and pasted

## Troubleshooting

**macOS**
- No response to fn: Check Accessibility permission, restart app
- Icon not visible: System Settings → Privacy & Security → Menu Bar → enable python
- Logs: `tail -f /tmp/groq-whisper.log`

**Windows**
- App not appearing in tray: Check `%USERPROFILE%\groq-whisper.log`
- Hotkey conflict: Set `FREE_WISPR_HOTKEY` to a different key combo
- Mic not working: Check Windows Settings → Privacy → Microphone
