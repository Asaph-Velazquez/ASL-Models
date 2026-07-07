# Mix Atencion

Pipeline de entrenamiento y evaluacion con TensorFlow para modelos con atencion.

## Requisitos

- Docker Desktop
- `data/raw/dataset.csv`

## Estructura esperada

- `config/`: configuracion del entrenamiento
- `data/raw/`: CSV de entrada
- `data/processed/`: datos serializados y label encoder
- `dataset/`: codigo de carga y preprocesamiento
- `model/`: arquitecturas, entrenamiento, prediccion y checkpoints
- `output/`: graficas y reportes

## Construir la imagen

```powershell
cd "mix atencion"
docker build -t sign-recognition .
```

## Ejecutar con Docker Compose

```powershell
cd "mix atencion"
docker compose up --build
```

## Ejecutar con el script de Windows

```powershell
cd "mix atencion"
run_docker.bat
```

## Salidas

- `data/processed/*.npy`
- `data/processed/label_encoder.pkl`
- `model/best_model.keras`
- `model/checkpoints/best_model.keras`
- `output/confusion_matrix.png`
- `output/results.json`
- `output/metrics_report.txt`
- `output/metrics_report.md`

## Notas

- El contenedor ejecuta `train_and_evaluate.py` y termina.
- El flujo actual espera que el CSV este en `data/raw/dataset.csv`.
- Si cambias el dataset, revisa `config/config.yaml` antes de volver a correr.
