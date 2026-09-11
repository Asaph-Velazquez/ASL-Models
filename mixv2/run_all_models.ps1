param(
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$projectRoot = $PSScriptRoot
$models = @("lstm_attention", "cnn_lstm", "svm")
$artifactBase = Join-Path $projectRoot "artifacts"
$resultsBase = Join-Path $projectRoot "results"

New-Item -ItemType Directory -Force -Path $artifactBase | Out-Null
New-Item -ItemType Directory -Force -Path $resultsBase | Out-Null

Set-Location $projectRoot

if (-not $NoBuild) {
    docker compose build sign-recognition
}

$jobs = @()
foreach ($model in $models) {
    $jobs += Start-Job -Name $model -ArgumentList $model, $projectRoot, $artifactBase -ScriptBlock {
        param($model, $projectRoot, $artifactBase)

        Set-Location $projectRoot
        $artifactRoot = Join-Path $artifactBase $model
        New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

        $env:MODEL_TYPE_OVERRIDE = $model
        $env:ARTIFACT_ROOT = "/app/artifacts/$model"

        docker compose run --rm `
            -e MODEL_TYPE_OVERRIDE=$model `
            -e ARTIFACT_ROOT=/app/artifacts/$model `
            sign-recognition 2>&1 |
            Tee-Object -FilePath (Join-Path $artifactRoot "run.log")
    }
}

Wait-Job $jobs | Out-Null

foreach ($job in $jobs) {
    Receive-Job $job
    Remove-Job $job
}

Write-Host "All model runs completed."
