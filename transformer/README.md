# Transformer

Modelo de lenguaje de senas basado en PyTorch Transformer.

## Requisitos

- Docker Desktop
- `data/dataset_35.csv`

## Estructura esperada

- `data/`: CSV de entrada
- `models/`: modelo entrenado
- `output/`: graficas y reportes

## Construir la imagen

```powershell
cd transformer
docker build -t sign-transformer .
```

## Ejecutar con Docker Compose

```powershell
cd transformer
docker compose up --build
```

## Ejecutar con el script de Windows

```powershell
cd transformer
run_docker.bat
```

## Salidas

- `output/training_history.png`
- `output/confusion_matrix.png`
- `output/confusion_matrix.csv`
- `output/results.json`
- `models/best_model.pt`

## Notas

- El contenedor ejecuta `train_and_evaluate.py` y termina.
- Si cambias el CSV, reconstruye o vuelve a ejecutar el contenedor con el mismo volumen `data/`.
