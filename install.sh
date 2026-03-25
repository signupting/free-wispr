#!/bin/bash
set -e

echo "=== Free Wispr Installer ==="
echo ""

# --- Find Python 3 ---
PYTHON=""
for candidate in python3 /usr/local/bin/python3 /opt/homebrew/bin/python3 "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3"; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3 not found. Install it first:"
    echo "  brew install python3"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# --- Install dependencies ---
echo ""
echo "Installing Python dependencies..."
"$PYTHON" -m pip install --quiet numpy scipy sounddevice groq pyobjc-framework-Cocoa requests

# --- API keys ---
echo ""
APP_DIR="$HOME/.local/groq-whisper-app"
ENV_FILE="$APP_DIR/.env"
mkdir -p "$APP_DIR"

if [ -f "$ENV_FILE" ]; then
    echo "Found existing .env file, loading..."
    source "$ENV_FILE"
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "You need a Groq API key (free at https://console.groq.com/keys)"
    read -rp "Enter your GROQ_API_KEY: " GROQ_API_KEY
    if [ -z "$GROQ_API_KEY" ]; then
        echo "ERROR: GROQ_API_KEY is required."
        exit 1
    fi
fi

if [ -z "$HF_API_KEY" ]; then
    echo ""
    echo "Optional: HuggingFace API key for fallback (press Enter to skip)"
    read -rp "Enter your HF_API_KEY (optional): " HF_API_KEY
fi

# Save keys
cat > "$ENV_FILE" << EOF
GROQ_API_KEY=$GROQ_API_KEY
HF_API_KEY=$HF_API_KEY
EOF
chmod 600 "$ENV_FILE"
echo "API keys saved to $ENV_FILE"

# --- Copy app ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/groq_whisper.py" "$APP_DIR/groq_whisper.py"
mkdir -p "$APP_DIR/backups"

# --- Create .app bundle ---
APP_PATH="/Applications/Free Wispr.app"
echo ""
echo "Creating app bundle at $APP_PATH..."

mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Info.plist
cat > "$APP_PATH/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.local.free-wispr</string>
    <key>CFBundleName</key>
    <string>Free Wispr</string>
    <key>CFBundleDisplayName</key>
    <string>Free Wispr</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Free Wispr needs microphone access for speech-to-text dictation.</string>
</dict>
</plist>
PLIST

# Launcher script
cat > "$APP_PATH/Contents/MacOS/launcher" << LAUNCHER
#!/bin/bash
source "$APP_DIR/.env"
export GROQ_API_KEY
export HF_API_KEY
exec "$PYTHON" "$APP_DIR/groq_whisper.py"
LAUNCHER
chmod +x "$APP_PATH/Contents/MacOS/launcher"

echo "App bundle created."

# --- Launch ---
echo ""
echo "Launching Free Wispr..."
open "$APP_PATH"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "IMPORTANT - Grant these permissions in System Settings > Privacy & Security:"
echo "  1. Microphone    → Free Wispr (required)"
echo "  2. Accessibility → Free Wispr (required for fn key detection)"
echo ""
echo "Usage: Tap the fn key to start/stop dictation."
echo "The microphone icon appears in your menu bar."
echo ""
echo "To uninstall: ./uninstall.sh"
