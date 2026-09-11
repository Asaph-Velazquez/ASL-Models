import json
import time
from pathlib import Path

import numpy as np


def _stats(arr):
    if arr is None:
        return None
    try:  # torch.Tensor -> numpy sin importar torch aqui
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
    except Exception:
        return {"tipo": type(arr).__name__, "repr": str(arr)[:120]}
    if arr.dtype.kind not in "fiub":
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    flat = arr.astype(np.float64).ravel()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(flat)) if flat.size else None,
        "max": float(np.nanmax(flat)) if flat.size else None,
        "mean": float(np.nanmean(flat)) if flat.size else None,
        "nan_count": int(np.isnan(flat).sum()),
    }


def _sample(arr, k):
    if arr is None:
        return None
    try:
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        flat = np.asarray(arr).ravel()[:k]
        return [round(float(v), 5) for v in flat.tolist()]
    except Exception:
        return None


class _Stage:
    def __init__(self, tracer, name, description):
        self.tracer = tracer
        self.name = name
        self.description = description
        self.record = {"capa": name, "descripcion": description, "metricas": {}}
        self._t0 = None

    def io(self, inp=None, out=None):
        self.record["entrada"] = _stats(inp)
        self.record["salida"] = _stats(out)
        self.record["muestra_salida"] = _sample(out, self.tracer.max_sample_values)

    def metric(self, key, value):
        self.record["metricas"][key] = value

    def note(self, text):
        self.record.setdefault("notas", []).append(str(text))

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.record["tiempo_s"] = round(time.perf_counter() - self._t0, 4)
        if exc is not None:
            self.record["error"] = f"{exc_type.__name__}: {exc}"
        self.tracer._add(self.record)
        return False


class TraceRecorder:
    def __init__(self, enabled=True, reports_dir="reports", max_sample_values=6, echo=True):
        self.enabled = enabled
        self.reports_dir = Path(reports_dir)
        self.max_sample_values = max_sample_values
        self.echo = echo
        self.records = []

    def stage(self, name, description=""):
        if not self.enabled:
            return _NullStage()
        return _Stage(self, name, description)

    def _add(self, record):
        self.records.append(record)
        if self.echo:
            print(self._format_record(record))

    @staticmethod
    def _format_record(r):
        lines = [f"[TRAZA] {r['capa']} — {r['descripcion']}  ({r.get('tiempo_s', '?')} s)"]
        for k in ("entrada", "salida"):
            s = r.get(k)
            if s:
                extra = ""
                if s.get("mean") is not None:
                    extra = (f" min={s['min']:.4g} max={s['max']:.4g}"
                             f" media={s['mean']:.4g} NaN={s['nan_count']}")
                lines.append(f"    {k}: shape={s.get('shape')} dtype={s.get('dtype')}{extra}")
        if r.get("muestra_salida"):
            lines.append(f"    muestra -> siguiente capa: {r['muestra_salida']}")
        for k, v in r.get("metricas", {}).items():
            lines.append(f"    metrica {k}: {v}")
        for n in r.get("notas", []):
            lines.append(f"    nota: {n}")
        if "error" in r:
            lines.append(f"    ERROR: {r['error']}")
        return "\n".join(lines)

    def save(self, basename="pipeline_trace"):
        if not self.enabled or not self.records:
            return
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / f"{basename}.json").write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False))
        txt = "\n\n".join(self._format_record(r) for r in self.records)
        (self.reports_dir / f"{basename}.txt").write_text(txt)
        print(f"[TRAZA] guardada en {self.reports_dir}/{basename}.json y .txt")


class _NullStage:
    def io(self, inp=None, out=None):
        pass

    def metric(self, key, value):
        pass

    def note(self, text):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
