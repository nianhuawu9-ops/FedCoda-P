Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$System = Join-Path $Root "third_party\PFLlib\system"

$Data = if ($args.Count -ge 1) { $args[0] } else { "Cifar10" }
$Rounds = if ($args.Count -ge 2) { $args[1] } else { "50" }
$Device = if ($args.Count -ge 3) { $args[2] } else { "cuda" }
$Algorithms = @("FedCoda-P", "FedRep", "FedPer")

Push-Location $System
try {
    foreach ($Algo in $Algorithms) {
        Write-Host "Running $Algo on $Data for $Rounds rounds"
        python main.py -data $Data -m CNN -algo $Algo -gr $Rounds -jr 0.2 -ls 1 -pls 1 -nc 20 -did 0 -dev $Device
        if ($LASTEXITCODE -ne 0) {
            throw "$Algo failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
