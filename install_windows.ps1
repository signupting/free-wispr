# Free Wispr Windows Installer
# Run with: powershell -ExecutionPolicy Bypass -File install_windows.ps1

Write-Host "=== Free Wispr Windows Installer ===" -ForegroundColor Cyan
Write-Host ""

# --- Find Python ---
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3") {
            $python = $candidate
            break
        }
    } catch {}
}

if (-not $python) {
    Write-Host "ERROR: Python 3 not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ (check 'Add to PATH')"
    exit 1
}

Write-Host "Using: $python ($( & $python --version 2>&1 ))"

# --- Install dependencies ---
Write-Host ""
Write-Host "Installing dependencies..."
& $python -m pip install --quiet numpy scipy sounddevice groq requests pystray pillow keyboard pyperclip

# --- API keys ---
Write-Host ""
$appDir = "$env:USERPROFILE\.free-wispr"
$envFile = "$appDir\.env"
New-Item -ItemType Directory -Force -Path $appDir | Out-Null

$groqKey = ""
$hfKey = ""

if (Test-Path $envFile) {
    Write-Host "Found existing config, loading..."
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^GROQ_API_KEY=(.+)") { $groqKey = $matches[1] }
        if ($_ -match "^HF_API_KEY=(.+)") { $hfKey = $matches[1] }
    }
}

if (-not $groqKey) {
    Write-Host "You need a Groq API key (free at https://console.groq.com/keys)"
    $groqKey = Read-Host "Enter your GROQ_API_KEY"
    if (-not $groqKey) {
        Write-Host "ERROR: GROQ_API_KEY is required." -ForegroundColor Red
        exit 1
    }
}

if (-not $hfKey) {
    Write-Host ""
    Write-Host "Optional: HuggingFace API key for fallback (press Enter to skip)"
    $hfKey = Read-Host "Enter your HF_API_KEY (optional)"
}

# Save config
@"
GROQ_API_KEY=$groqKey
HF_API_KEY=$hfKey
"@ | Set-Content $envFile
Write-Host "Config saved to $envFile"

# --- Copy script ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$scriptDir\groq_whisper_windows.py" "$appDir\groq_whisper_windows.py" -Force
New-Item -ItemType Directory -Force -Path "$appDir\backups" | Out-Null

# --- Create launcher .bat ---
$launcher = "$appDir\launch.bat"
@"
@echo off
set GROQ_API_KEY=$groqKey
set HF_API_KEY=$hfKey
$python "$appDir\groq_whisper_windows.py"
"@ | Set-Content $launcher

# --- Add to startup ---
$startupDir = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
$shortcutPath = "$startupDir\Free Wispr.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WindowStyle = 7  # minimized
$shortcut.Description = "Free Wispr speech-to-text"
$shortcut.Save()
Write-Host "Added to Windows startup"

# --- Launch ---
Write-Host ""
Write-Host "Launching Free Wispr..."
Start-Process -FilePath $launcher -WindowStyle Hidden

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Free Wispr is running in your system tray (bottom-right)."
Write-Host "Press Ctrl+Shift+Space to start/stop dictation."
Write-Host ""
Write-Host "To use a different hotkey, set the FREE_WISPR_HOTKEY environment variable."
Write-Host "Example: ctrl+alt+space, right ctrl, f9, etc."
