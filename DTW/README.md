# DTW

Este modulo contiene el script de entrenamiento y comparacion para el modelo DTW/SVM.
El procesamiento del CSV detecta automaticamente el separador y soporta dos esquemas de features:

- `kpca_*`
- landmarks `l0_x` ... `l20_z`

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

Este es el modo por defecto. Ejecuta SVM y DTW sobre el mismo split y guarda los modelos en `DTW/models`.

```powershell
cd DTW
New-Item -ItemType Directory -Force models | Out-Null
docker run --rm -v "${PWD}\models:/app/models" dtw-model compare
```

## Otros modos

### Entrenar un modelo

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model train --model svm
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model train --model dtw
```

### Evaluar un modelo

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model evaluate --model svm
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model evaluate --model dtw
```

### Inferencia

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model infer --model svm --input datos_30.csv
```

```powershell
docker run --rm -v "${PWD}\models:/app/models" dtw-model infer --model dtw --input datos_30.csv
```

## Salidas

Los artefactos se guardan en `DTW/models`:

- `model_svm.joblib`
- `model_dtw.joblib`

## Notas

- El contenedor no queda levantado como servicio; ejecuta el script y termina.
- Si cambias el CSV, asegurate de usar el mismo esquema de features para entrenar e inferir.
