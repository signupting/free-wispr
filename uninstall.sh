#!/bin/bash
echo "=== Uninstalling Free Wispr ==="

# Kill running process
pkill -f "groq_whisper.py" 2>/dev/null && echo "Stopped running process." || true

# Remove app bundle
if [ -d "/Applications/Free Wispr.app" ]; then
    rm -rf "/Applications/Free Wispr.app"
    echo "Removed /Applications/Free Wispr.app"
fi

# Remove app data
if [ -d "$HOME/.local/groq-whisper-app" ]; then
    read -rp "Remove app data including API keys? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$HOME/.local/groq-whisper-app"
        echo "Removed ~/.local/groq-whisper-app"
    else
        echo "Kept ~/.local/groq-whisper-app (contains your API keys)"
    fi
fi

echo ""
echo "Free Wispr has been uninstalled."
echo "You may also want to remove it from System Settings > Privacy & Security permissions."
