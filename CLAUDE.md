# Free Wispr — Claude Code Install Guide

## Install for user

```bash
# Clone the repo
git clone https://github.com/signupting/free-wispr.git /tmp/free-wispr-install
cd /tmp/free-wispr-install

# Run the installer (will prompt for API key)
bash install.sh
```

## What the installer does

1. Finds Python 3 on the system
2. Installs deps: `numpy scipy sounddevice groq pyobjc-framework-Cocoa requests`
3. Prompts for GROQ_API_KEY (required) and HF_API_KEY (optional)
4. Copies app to `~/.local/groq-whisper-app/`
5. Creates `/Applications/Free Wispr.app` bundle
6. Launches the app

## After install — user must do manually

1. **System Settings > Privacy & Security > Microphone** → enable Free Wispr
2. **System Settings > Privacy & Security > Accessibility** → enable Free Wispr
3. Restart the app after granting permissions

## Get a Groq API key

Free at https://console.groq.com/keys

## Uninstall

```bash
cd /tmp/free-wispr-install && bash uninstall.sh
```

## Troubleshooting

- Logs: `tail -f /tmp/groq-whisper.log`
- Kill: `pkill -f groq_whisper.py`
- Relaunch: `open "/Applications/Free Wispr.app"`
