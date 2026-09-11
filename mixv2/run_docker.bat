@echo off
echo Construyendo imagen Docker...
docker-compose build

echo.
echo Ejecutando entrenamiento...
docker-compose up


echo Para detener: docker-compose down
pause