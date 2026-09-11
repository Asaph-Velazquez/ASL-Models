
## Que revisar antes

Comprobación de que Docker ve la GPU:
   ```powershell
   docker run --rm --gpus all pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime nvidia-smi
   ```
   Debe imprimir la tabla de `nvidia-smi`. Si falla, actualiza driver y Docker Desktop.

**Construir la imagen (una sola vez, y cada vez que cambie el código):**
```bash
docker compose build
```


## Orden de los datos

```
Dataset/
  Aisladas/                      <- palabras aisladas: carpeta = glosa
    CAR/ 01.mp4 02.mp4 ...
    STAY/ ...  USE/ ...  WITH/ ...  (una carpeta por glosa del vocabulario)
    PLACE/ V1/01.mp4 ...           (se aceptan subcarpetas de variantes)
    NONE/  OTHER/                  (señas reales de contenido, clases normales)
    FS-J/  FS-Z/                   (letras DINÁMICAS en video)
    letras/
      Train_Alphabet/ A/ B/ Blank/ ... Z/   (imágenes .png)
      Test_Alphabet/  A/ B/ Blank/ ... Z/
    numbers/
      Train_Nums/ 0/ 1/ ... 9/    (imágenes)
      Test_Nums/  0/ 1/ ... 9/
  Frases/                        <- frases continuas para el CTC
    "ROOM SHEETS I WANT CHANGE.mp4"   (el NOMBRE es la secuencia de glosas)
    annotations.csv                    (opcional/recomendado, ver abajo)
asl_hotel_vocabulary.csv
signer_manifest.csv
external/
  how2sign/                      <- lo llena el servicio `how2sign` (descarga de internet)
    csv/  filtered_*.csv  video_clips/{train,val,test}/  how2sign_manifest.csv
```

## Pasos a realizar si todo ya esta configurado

### Paso 0 — construir la imagen
```bash
docker compose build
```

### Paso 1 — smoke test (siempre, antes que nada)
```bash
docker compose run --rm smoke
```
Corre TODO el pipeline con datos falsos en segundos. Si termina con `SMOKE TEST OK`, la estructura encaja. Ver `docs/COMO_PROBAR.md`.

### Paso 1-bis — How2Sign: descargar y mapear (solo porque la estructura actual no los contiene directamente)
```bash
docker compose run --rm how2sign
```
Descarga de Google Drive (IDs oficiales de how2sign.github.io) los CSVs de traducción inglesa realineada y los **clips frontales por oración**, filtra las oraciones contra tu vocabulario (pseudo-glosas; ver abajo), extrae SOLO los clips aceptados a `external/how2sign/video_clips/` y **anexa el mapeo a `signer_manifest.csv`** (idempotente: correrlo dos veces no duplica).

- **Tamaños**: CSVs son KBs; los zips de clips pesan **train 31G / val 1.7G / test 2.2G**. El zip se borra tras extraer (usa `--keep-zip` para conservarlo).
- **Por partes** (recomendado para probar primero sin el zip de 31G):
  ```bash
  docker compose run --rm how2sign python data/download_how2sign.py --config configs/how2sign.yaml --csv-only
  docker compose run --rm how2sign python data/download_how2sign.py --config configs/how2sign.yaml --splits val test
  docker compose run --rm how2sign python data/download_how2sign.py --config configs/how2sign.yaml --splits train
  docker compose run --rm how2sign python data/map_how2sign_manifest.py --config configs/how2sign.yaml
  ```

### Paso 2 — validar y preprocesar (en la máquina Windows con GPU)
```bash
docker compose run --rm preprocess
```
Qué hace: valida vocabulario/manifiestos/frases (reporte en `reports/validation_report.json`), evalúa la calidad de cada video y aplica filtros automáticos si hacen falta (Capa 0-bis), extrae landmarks con MediaPipe (manos + torso, SIN cara) y cachea todo como `.npy` en `outputs/cache/`. Incluye los clips How2Sign si corriste el Paso 1-bis (sus splits oficiales train/val/test se respetan, no se recalculan). También decide los **splits**: signer-independent con `signer_manifest.csv`; letras/números usan las carpetas Train_/Test_ provistas.
Duración: minutos a horas según número de videos e imágenes (es el paso más pesado junto con las fases). Produce: `outputs/cache/{landmarks/, phrases/, how2sign/, clips_manifest.csv, phrases_manifest.csv, how2sign_phrases_manifest.csv, layout.json, preprocess_log.json}`.
**Lee los AVISOS que imprime**: frases rechazadas, carpetas excluidas, clases sin datos.

### Paso 3 — generar frases sintéticas + corpus paralelo
```bash
docker compose run --rm synthetic
```
Concatena clips aislados en frases continuas con orden ASL por ramas hoteleras (volumen para el CTC) y genera el corpus paralelo `(glosas, inglés)` para la Fase 4. Minutos. Produce `outputs/synthetic/`. El split `test` sintético usa SOLO clips de firmantes de test (métrica honesta).

### Paso 4 — Fase 1: pre-entreno con letras/números
```bash
docker compose run --rm train python train/train_phase1.py --config configs/phase1.yaml
```
Entrena encoder + clasificación solo con las imágenes de letras/números (mini-secuencias). Da un encoder inicial barato. ~minutos-horas según GPU. Produce `checkpoints/phase1/{best.pt,last.pt,summary.json}`.

### Paso 5 — Fase 2: clasificación aislada, vocabulario completo
```bash
docker compose run --rm train python train/train_phase2.py --config configs/phase2.yaml
```
Inicializa el encoder desde la Fase 1 y entrena con TODOS los aislados (videos + imágenes). `FS-J`/`FS-Z` entrenan con AMBAS fuentes (video dinámico + imagen estática). Produce `checkpoints/phase2/`.

### Paso 6 — Fase 3: CTC continuo (el puente aislado→continuo)
```bash
docker compose run --rm train python train/train_phase3.py --config configs/phase3.yaml
```
Reusa el encoder de la Fase 2. Entrena CTC con frases sintéticas (volumen) + frases reales de `Frases/` (coarticulación) + clips How2Sign (señado continuo real con pseudo-glosas, peso débil `how2sign_weight`, excluidos del currículo inicial y de val/test), con currículo (primeras épocas solo sintético) y peso extra a lo real. Produce `checkpoints/phase3/`. **Ojo:** el WER de validación durante el entrenamiento es sintético; el WER que vale es el de frases reales (lo da `eval`). No uses canciones en ASL como dato extra: su gramática lírica daña la transferencia al dominio.

### Paso 7 — Fase 4: traducción glosa→inglés (T5)
```bash
docker compose run --rm train python train/train_phase4.py --config configs/phase4.yaml
```
Afinado de `t5-small` (se descarga una vez a `external/hf_cache`) con el corpus paralelo. Produce `checkpoints/phase4/best/`. El inglés resultante está acotado a las plantillas del dominio hotelero.

### Paso 8 — evaluación
```bash
docker compose run --rm eval
```
Ver `docs/COMO_VER_RESULTADOS.md`.

### Todo de una vez (opcional)
```bash
docker compose run --rm train bash run_all.sh
```

## 5. Reanudar y ajustar

- **Si un entrenamiento se interrumpe**, vuelve a correr la MISMA línea: cada fase detecta `checkpoints/phaseN/last.pt` y reanuda sola desde la última época.
- **Cambiar hiperparámetros**: edita `configs/phaseN.yaml` (épocas, `batch_size`, `lr`, arquitectura `mlp|gcn`, `bilstm|tcn`, pesos de mezcla de la Fase 3...). Los configs están montados en el contenedor: no hace falta reconstruir la imagen.
- **Prueba rápida antes de entrenar en serio**: baja `epochs` a 2-3 en los YAML, o preprocesa un subconjunto con `docker compose run --rm train python data/preprocess.py --config configs/data.yaml --limit 50`.
- Semillas y flags deterministas ya están fijados en los configs (`seed`, `deterministic`).
