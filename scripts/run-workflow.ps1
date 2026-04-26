# Curated workflows for Windows (PowerShell). Repo root: run from project root.
# Examples:
#   .\scripts\run-workflow.ps1 validate
#   .\scripts\run-workflow.ps1 train-better
#   .\scripts\run-workflow.ps1 mlflow-better
#   $env:MLFLOW_TRACKING_URI = "file:./mlruns"; .\scripts\run-workflow.ps1 train-better

param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "setup", "quickstart", "check", "check-cov", "ci-local", "install", "install-train", "install-dev",
        "validate", "etl", "explain", "train-baseline", "train-better", "train-advanced", "train-xgb", "train-all",
        "test", "test-cov", "inference", "api", "mlflow-better", "download-better", "help"
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
    "setup" { Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt") }
    "quickstart" {
        Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt")
        Invoke-Py @("-m", "src.pipeline.validate")
        Invoke-Py @("-m", "src.train", "--model", "better", "--cv-f1")
        Invoke-Py @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
    }
    "check" {
        Invoke-Py @("-m", "src.pipeline.validate")
        Invoke-Py @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
    }
    "check-cov" {
        Invoke-Py @("-m", "src.pipeline.validate")
        Invoke-Py @("-m", "pip", "install", "-q", "coverage")
        Invoke-Py @("-m", "coverage", "run", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        Invoke-Py @("-m", "coverage", "report", "-m")
    }
    "ci-local" {
        Invoke-Py @("-m", "pip", "install", "-q", "-r", "requirements-ci.txt")
        Invoke-Py @("-m", "pip_audit", "-r", "requirements.txt", "--desc", "on")
        Invoke-Py @("-m", "pip_audit", "-r", "requirements_train.txt", "--desc", "on")
        Invoke-Py @("-m", "bandit", "-r", "src", "-ll", "-f", "txt")
        Invoke-Py @("-m", "coverage", "run", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        Invoke-Py @("-m", "coverage", "report", "-m", "--fail-under=70")
    }
    "install" { Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt") }
    "install-train" { Invoke-Py @("-m", "pip", "install", "-r", "requirements_train.txt") }
    "install-dev" { Invoke-Py @("-m", "pip", "install", "-r", "requirements-dev.txt") }
    "validate" { Invoke-Py @("-m", "src.pipeline.validate") }
    "etl" { Invoke-Py @("-m", "src.pipeline.etl") }
    "explain" { Invoke-Py @("-m", "src.interpretability") }
    "train-baseline" { Invoke-Py @("-m", "src.train", "--model", "baseline", "--cv-f1") }
    "train-better" { Invoke-Py @("-m", "src.train", "--model", "better", "--cv-f1") }
    "train-advanced" { Invoke-Py @("-m", "src.train", "--model", "advanced", "--cv-f1") }
    "train-xgb" { Invoke-Py @("-m", "src.train", "--model", "advanced_xgb", "--cv-f1") }
    "train-all" {
        Invoke-Py @("-m", "src.train", "--model", "baseline", "--cv-f1")
        Invoke-Py @("-m", "src.train", "--model", "better", "--cv-f1")
        Invoke-Py @("-m", "src.train", "--model", "advanced", "--cv-f1")
        Invoke-Py @("-m", "src.train", "--model", "advanced_xgb", "--cv-f1")
    }
    "test" { Invoke-Py @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py") }
    "test-cov" {
        Invoke-Py @("-m", "pip", "install", "-q", "coverage")
        Invoke-Py @("-m", "coverage", "run", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        Invoke-Py @("-m", "coverage", "report", "-m")
    }
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
        Write-Host "Common: setup, quickstart, check, check-cov, ci-local, install-dev, etl, explain, train-all"
        Write-Host "Also: install, install-train, validate, train-*, test, test-cov, inference, api, mlflow-better, download-better"
    }
}
