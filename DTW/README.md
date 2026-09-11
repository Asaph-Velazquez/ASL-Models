# DTW

Este modulo contiene el script de entrenamiento y comparacion para el modelo DTW/SVM.
El CSV `datos_30.csv` se adapta automaticamente y soporta dos esquemas de features:

- `kpca_*`
- landmarks `l0_x` ... `l20_z`

Para este dataset, la agrupacion de manos usa `handedness` (`Left` / `Right`) cuando existe.
Si el archivo no trae `handedness`, el script cae a `hand_index` como respaldo.

## Requisitos

- Docker Desktop
- El archivo `datos_30.csv` dentro de esta carpeta

## Construir la imagen

Desde la raiz del repositorio:

```powershell
cd DTW
docker build -t dtw-model .
```

## Ejecutar comparacion

Este es el modo por defecto. Ejecuta SVM y DTW sobre el mismo split, guarda los modelos en `DTW/models` y escribe las metricas en `DTW/results`.

```powershell
cd DTW
New-Item -ItemType Directory -Force models,results | Out-Null
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model compare
```

## Otros modos

### Entrenar un modelo

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model train --model svm
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model train --model dtw
```

### Evaluar un modelo

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model evaluate --model svm
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model evaluate --model dtw
```

### Inferencia

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model infer --model svm --input datos_30.csv
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" -v "${PWD}\results:/app/results" dtw-model infer --model dtw --input datos_30.csv
```

## Salidas

Los artefactos se guardan en `DTW/models`:

- `model_svm.joblib`
- `model_dtw.joblib`

Las metricas se guardan en `DTW/results`:

- `latest_compare.json` / `latest_compare.txt`
- `latest_evaluate_svm.json` / `latest_evaluate_svm.txt`
- `latest_evaluate_dtw.json` / `latest_evaluate_dtw.txt`
- `latest_train_svm.json` / `latest_train_svm.txt`
- `latest_train_dtw.json` / `latest_train_dtw.txt`

## Notas

- El contenedor no queda levantado como servicio; ejecuta el script y termina.
- El modelo guarda metadatos de `hand_column` y `hand_order` para que la inferencia use la misma representacion que el entrenamiento.
- Si cambias el CSV, asegurate de usar el mismo esquema de features para entrenar e inferir.
