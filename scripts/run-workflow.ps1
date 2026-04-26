# Curated workflows for Windows (PowerShell). Repo root: run from project root.
# Examples:
#   .\scripts\run-workflow.ps1 validate
#   .\scripts\run-workflow.ps1 train-better
#   .\scripts\run-workflow.ps1 mlflow-better
#   $env:MLFLOW_TRACKING_URI = "file:./mlruns"; .\scripts\run-workflow.ps1 train-better

param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "install", "install-train", "validate", "train-baseline", "train-better",
        "train-advanced", "train-xgb", "test", "inference", "api", "mlflow-better",
        "download-better", "help"
    )]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Invoke-Py {
    param([string[]]$Args)
    & $py @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Command) {
    "install" { Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt") }
    "install-train" { Invoke-Py @("-m", "pip", "install", "-r", "requirements_train.txt") }
    "validate" { Invoke-Py @("-m", "src.pipeline.validate") }
    "train-baseline" { Invoke-Py @("-m", "src.train", "--model", "baseline", "--cv-f1") }
    "train-better" { Invoke-Py @("-m", "src.train", "--model", "better", "--cv-f1") }
    "train-advanced" { Invoke-Py @("-m", "src.train", "--model", "advanced", "--cv-f1") }
    "train-xgb" { Invoke-Py @("-m", "src.train", "--model", "advanced_xgb", "--cv-f1") }
    "test" { Invoke-Py @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py") }
    "inference" {
        $m = if ($env:MODEL) { $env:MODEL } else { "better" }
        Invoke-Py @(
            "-m", "src.inference",
            "--input-csv", "data/raw/Womens Clothing E-Commerce Reviews.csv",
            "--model-path", "models/${m}_pipeline.joblib"
        )
    }
    "api" { Invoke-Py @("-m", "uvicorn", "src.api:app", "--reload", "--host", "127.0.0.1", "--port", "8000") }
    "mlflow-better" { Invoke-Py @("-m", "src.train", "--model", "better", "--cv-f1", "--mlflow") }
    "download-better" { Invoke-Py @("-m", "src.train", "--download-data", "--model", "better", "--cv-f1") }
    "help" {
        Write-Host "Usage: .\scripts\run-workflow.ps1 <command>"
        Write-Host "Commands: install, install-train, validate, train-baseline, train-better, train-advanced, train-xgb, test, inference, api, mlflow-better, download-better"
    }
}
