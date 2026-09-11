# Transformer

Modelo de lenguaje de senas basado en PyTorch Transformer.

## Requisitos

- Docker Desktop
- `data/dataset_35.csv`

## Esquema esperado del CSV

El pipeline actual detecta automaticamente las features numericas del archivo.
Para el dataset nuevo espera estas columnas de metadatos:

- `glosa`
- `video`
- `frame`
- `hand_index` y `handedness` son opcionales como metadatos

El resto de columnas numericas se usan como secuencia de entrada. Eso permite
cargar tanto el dataset nuevo de landmarks como una version anterior con
`kpca_*`.

## Estructura

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
- Si cambias el CSV, vuelve a construir o ejecuta de nuevo el contenedor con el volumen `data/` actualizado.
