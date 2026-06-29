Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$System = Join-Path $Root "third_party\PFLlib\system"

$Data = if ($args.Count -ge 1) { $args[0] } else { "Cifar100" }

$Algorithms = @("FedAvg", "FedRep", "FedALA", "FedPAC", "FedCP", "FedDBE", "FedAS")

Push-Location $System
try {
    foreach ($Algo in $Algorithms) {
        Write-Host "Running $Algo on $Data"
        python main.py -data $Data -m CNN -algo $Algo -gr 200 -jr 0.6 -ls 5 -pls 1 -nc 20 -did 0
    }
}
finally {
    Pop-Location
}
