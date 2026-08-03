#!/bin/zsh
# Launcher for Free Wispr, invoked by ~/Library/LaunchAgents/com.clouds.free-wispr.plist
# Keeps GROQ_API_KEY / HF_API_KEY in .env rather than in the plist.

APP_DIR="$HOME/.local/groq-whisper-app"
cd "$APP_DIR" || exit 1

# parakeet_mlx -> librosa and the wav path have historically wanted homebrew on PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Load .env without echoing it.
if [ -f "$APP_DIR/.env" ]; then
  set -a
  . "$APP_DIR/.env"
  set +a
fi

# The .app bundle python is required — NSStatusBar needs a bundled GUI interpreter.
exec "$HOME/anaconda3/python.app/Contents/MacOS/python" "$APP_DIR/groq_whisper.py"
