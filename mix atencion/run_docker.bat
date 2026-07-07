@echo off

if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "model" mkdir model
if not exist "model\checkpoints" mkdir model\checkpoints
if not exist "logs" mkdir logs
if not exist "output" mkdir output

if not exist "data\raw\dataset.csv" (
    echo ERROR: No se encuentra data\raw\dataset.csv
    echo Coloca tu archivo CSV en esa ubicacion
    pause
    exit /b 1
)

echo CSV encontrado: data\raw\dataset.csv

echo Construyendo imagen Docker...
docker compose build

echo.
echo Ejecutando entrenamiento...
docker compose up

echo Para detener: docker compose down
pause
