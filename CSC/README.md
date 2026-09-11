```
Video
 └─ Capa 0-bis filtrado adaptativo automático (brillo/nitidez/denoise/fps/zoom a manos)
 └─ Capa 0  MediaPipe Holistic (manos + pose superior, sin cara) -> cache .npy
 └─ Capa 1  Normalización 2 caminos (cuerpo | mano según functional_class) + pos/vel/acel
 └─ Capa 2  Encoder espacial por frame (MLP; GCN opcional)
 └─ Capa 3  Encoder temporal (Bi-LSTM; TCN opcional)
 ├─ Capa 4A Clasificación N=len(vocab)+especiales (Fase 1: letras/números; Fase 2: todo)
 ├─ Capa 4B CTC sobre frases (Fase 3: sintético + frases reales)
 └─ Capa 5  Beam search CTC + n-grama de glosas        ---- SALIDA 1: GLOSAS
 └─ Capa 6  Traducción glosa→inglés (T5 + plantillas)  ---- SALIDA 2: INGLÉS  (Fase 4) no funciona del todo

docker compose run --rm how2sign       # (opcional, INTERNET) descarga How2Sign + mapeo
docker compose run --rm preprocess     # validar + landmarks -> cache (incluye How2Sign)
docker compose run --rm synthetic      # frases sintéticas + corpus paralelo
docker compose run --rm train python train/train_phase1.py --config configs/phase1.yaml
docker compose run --rm train python train/train_phase2.py --config configs/phase2.yaml
docker compose run --rm train python train/train_phase3.py --config configs/phase3.yaml
docker compose run --rm train python train/train_phase4.py --config configs/phase4.yaml
docker compose run --rm eval           # métricas + reportes en ./reports
docker compose run --rm demo python demo/infer.py --video <ruta>   # inferencia
docker compose up tensorboard          # http://localhost:6006
# o todo encadenado:
docker compose run --rm train bash run_all.sh

```

## How2Sign (fuente extra de internet para el CTC) que puede que funcione o no


- `docs/COMO_ENTRENAR.md` — requisitos, dónde van los datos, cada paso con su comando.
- `docs/COMO_VER_RESULTADOS.md` — dónde está cada métrica y cómo interpretarla.
- `docs/COMO_PROBAR.md` — smoke test, demo con video propio, checklist de errores.
