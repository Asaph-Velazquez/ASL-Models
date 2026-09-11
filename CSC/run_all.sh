#!/usr/bin/env bash
# Pipeline completo (validar -> preprocesar -> sintetico -> Fases 1-4 -> eval): docker compose run --rm train bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== [1/8] Validacion de datos =="
python data/validate_manifest.py --config configs/data.yaml

echo "== [2/8] Preprocesamiento (landmarks -> cache .npy) =="
python data/preprocess.py --config configs/data.yaml

echo "== [3/8] Generacion sintetica + corpus paralelo =="
python synthetic/generator.py --config configs/synthetic.yaml

echo "== [4/8] Fase 1: pre-entreno letras/numeros (imagenes) =="
python train/train_phase1.py --config configs/phase1.yaml

echo "== [5/8] Fase 2: clasificacion aislada (vocabulario completo) =="
python train/train_phase2.py --config configs/phase2.yaml

echo "== [6/8] Fase 3: CTC continuo (sintetico + frases reales) =="
python train/train_phase3.py --config configs/phase3.yaml

echo "== [7/8] Fase 4: traduccion glosa->ingles (T5) =="
python train/train_phase4.py --config configs/phase4.yaml

echo "== [8/8] Evaluacion =="
python eval/metrics.py --config configs/eval.yaml

echo "Pipeline completo. Reportes en ./reports, checkpoints en ./checkpoints."
