@echo off

if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "output" mkdir output

if not exist "data\dataset_35.csv" (
    echo ERROR: No se encuentra data\dataset_35.csv
    echo Coloca tu archivo CSV en esa ubicacion
    pause
    exit /b 1
)

echo CSV encontrado: data\dataset_35.csv

echo.
echo Construyendo imagen y ejecutando entrenamiento...
docker compose up --build --abort-on-container-exit

echo.
echo ==============================================
echo PROCESO COMPLETADO
echo ==============================================
echo Resultados en:
echo   - output\training_history.png
echo   - output\confusion_matrix.png
echo   - output\confusion_matrix.csv
echo   - output\results.json
echo   - models\best_model.pt
pause
