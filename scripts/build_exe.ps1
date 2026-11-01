# Build the standalone Windows executable.
# Requires: python -m pip install pyinstaller
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "1/3 Building the web frontend…"
Push-Location web
npm run build
Pop-Location

Write-Host "2/3 Preparing the build directory…"
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist-exe) { Remove-Item -Recurse -Force dist-exe }

Write-Host "3/3 Running PyInstaller…"
$webDist = Join-Path $root "web\dist"
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name HCSFoundry `
  --distpath dist-exe `
  --workpath build/pyinstaller `
  --specpath build `
  --add-data "$webDist;web/dist" `
  --collect-submodules uvicorn `
  --collect-submodules fastapi `
  --collect-submodules starlette `
  --hidden-import uvicorn.logging `
  --hidden-import uvicorn.loops.auto `
  --hidden-import uvicorn.protocols.http.auto `
  --hidden-import uvicorn.protocols.websockets.auto `
  --hidden-import uvicorn.lifespan.on `
  --hidden-import anyio._backends._asyncio `
  --exclude-module torch `
  --exclude-module transformers `
  --exclude-module peft `
  --exclude-module datasets `
  --exclude-module matplotlib `
  --exclude-module tkinter `
  server/foundry/launcher.py

if (-not (Test-Path "dist-exe/HCSFoundry.exe")) {
  throw "Build failed: dist-exe/HCSFoundry.exe not found"
}

$size = [math]::Round((Get-Item "dist-exe/HCSFoundry.exe").Length / 1MB, 1)
Write-Host "Built dist-exe/HCSFoundry.exe ($size MB)"
