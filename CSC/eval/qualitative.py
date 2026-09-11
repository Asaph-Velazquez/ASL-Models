"""tabla copiada: glosa_real | glosa_pred | ingles_pred | ingles_ref (CSV + Markdown)."""
import csv
from pathlib import Path


def write_qualitative_table(rows, out_dir, name="qualitative_table"):
    """rows: dicts con glosa_real, glosa_pred, ingles_pred, ingles_ref."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["glosa_real", "glosa_pred", "ingles_pred", "ingles_ref"]
    csv_path = out_dir / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])

    md_path = out_dir / f"{name}.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "---|" * len(fields) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")).replace("|", "\\|")
                                      for k in fields) + " |\n")
    return csv_path, md_path
