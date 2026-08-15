param(
    [switch]$StartServices,
    [switch]$StartServer,
    [switch]$StartWorkers,
    [switch]$InstallDeps
)

# Setup script for backend_ocr (Windows PowerShell)
# Usage examples:
#   .\setup.ps1                # create venv, install deps, run migrations
#   .\setup.ps1 -StartServices # also launches Django server, Celery worker and beat in separate windows
#   .\setup.ps1 -InstallDeps:$false  # skip installing deps

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Write-Host "[setup] Running from: $scriptDir"

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python is not found in PATH. Install Python 3.10+ and re-run this script."
    exit 1
}

$venvDir = Join-Path $scriptDir '.venv'
$pythonExe = Join-Path $venvDir 'Scripts\python.exe'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'

# Create virtualenv if missing
if (-not (Test-Path $venvDir)) {
    Write-Host "[setup] Creating virtual environment at $venvDir"
    python -m venv $venvDir
}

# Install dependencies
if ($InstallDeps -or (-not $PSBoundParameters.ContainsKey('InstallDeps'))) {
    Write-Host "[setup] Installing/Updating pip and requirements"
    & $pythonExe -m pip install --upgrade pip setuptools wheel
    & $pythonExe -m pip install -r requirements.txt
}

# Create .env if missing with recommended defaults
$envPath = Join-Path $scriptDir '.env'
if (-not (Test-Path $envPath)) {
    Write-Host "[setup] Creating .env with recommended defaults (please review and edit)"
    $envText = @"
# Django settings
DJANGO_DEBUG=True
SECRET_KEY=replace_me_with_secure_value
# Celery/Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
# Optional: set to full HuggingFace repo or local path
BASE_MODEL_NAME=sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3
"@
    $envText | Out-File -FilePath $envPath -Encoding UTF8
}

# Run migrations
Write-Host "[setup] Running Django migrations"
& $pythonExe manage.py makemigrations --noinput
& $pythonExe manage.py migrate --noinput

# Collect static (optional)
try {
    & $pythonExe manage.py collectstatic --noinput
} catch {
    Write-Warning "collectstatic failed or not required: $_"
}

Write-Host "[setup] Setup steps completed."

if ($StartServices -or $StartServer -or $StartWorkers) {
    Write-Host "[setup] Starting selected services in new PowerShell windows..."
    # Server
    if ($StartServices -or $StartServer) {
        $cmd = "& `"$activateScript`"; python manage.py runserver 0.0.0.0:8000"
        Start-Process powershell -ArgumentList '-NoExit','-Command',$cmd -WorkingDirectory $scriptDir
        Start-Sleep -Milliseconds 500
    }
    # Celery worker
    if ($StartServices -or $StartWorkers) {
        $cmd = "& `"$activateScript`"; celery -A core worker --loglevel=info"
        Start-Process powershell -ArgumentList '-NoExit','-Command',$cmd -WorkingDirectory $scriptDir
        Start-Sleep -Milliseconds 500

        # Celery beat
        $cmd = "& `"$activateScript`"; celery -A core beat --loglevel=info"
        Start-Process powershell -ArgumentList '-NoExit','-Command',$cmd -WorkingDirectory $scriptDir
        Start-Sleep -Milliseconds 500
    }
    Write-Host "[setup] Services launched. Check the new windows for logs."
} else {
    Write-Host "[setup] To start services, re-run with -StartServices or -StartServer/-StartWorkers"
}

Write-Host "[setup] Done."
