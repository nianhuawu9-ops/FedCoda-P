Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$PFLlib = Join-Path $Root "third_party\PFLlib"

if (-not (Test-Path (Join-Path $PFLlib ".git"))) {
    git clone --depth 1 https://github.com/TsingZ0/PFLlib.git $PFLlib
}

git -C $PFLlib rev-parse --short HEAD
if (-not (Test-Path (Join-Path $PFLlib "system\flcore\servers\servercoda.py"))) {
    git -C $PFLlib apply --check (Join-Path $Root "fedcoda\patches\fedcoda_p_pfllib.patch")
    git -C $PFLlib apply (Join-Path $Root "fedcoda\patches\fedcoda_p_pfllib.patch")
}
Write-Host "PFLlib ready at $PFLlib"
