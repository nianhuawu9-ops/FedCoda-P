Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$System = Join-Path $Root "third_party\PFLlib\system"

Push-Location $System
try {
    python main.py -data Cifar100 -m CNN -algo FedCoda-P -gr 200 -jr 0.6 -ls 5 -pls 1 -nc 20 -did 0
}
finally {
    Pop-Location
}
