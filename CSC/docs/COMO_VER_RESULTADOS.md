# en donde aparecen los resultados de eval

Esto se verifica desde la carpeta de `./reports/`. Se genera con:
```bash
docker compose run --rm eval
```

## explicacion de las metricas

| Archivo | Métrica | Qué mide | Qué es "bueno" |
|---|---|---|---|
| `reports/metrics_summary.json` → `aislado` | accuracy top-1 / top-5 | % de señas aisladas bien clasificadas (top-5: la correcta está entre las 5 mejores) | top-1 > 0.85 en test signer-independent es sólido; letras confundibles (A/S/T/E) bajan esto |
| `metrics_summary.json` → `continuo.synthetic.wer` | WER glosas (sintético) | errores de palabra (sustituciones+inserciones+borrados)/longitud en frases sintéticas de firmantes NO vistos | menor es mejor; 0.0 = perfecto, 1.0 = todo mal |
| `metrics_summary.json` → `continuo.real.wer` | **WER glosas (frases reales)** | lo mismo sobre `Frases/` reales | **esta es LA métrica del sistema**; espera que sea peor que la sintética |
| `metrics_summary.json` → `continuo.brecha_sintetico_a_real` | brecha sintético→real | `wer_real − wer_sintético` | cuanto más chica, mejor transfiere; si es enorme, faltan frases reales de entrenamiento |
| `metrics_summary.json` → `traduccion` | BLEU / ROUGE-L / exact-match | calidad del inglés vs. referencia (BLEU 0-100; ROUGE-L y EM 0-1) | en dominio cerrado con plantillas, BLEU > 60 y EM alto son alcanzables; NO mide inglés libre |



## generacion de matriz de confusión — `reports/confusion_matrix.png`

## vista por capa -----> `reports/pipeline_trace.*`

`pipeline_trace.txt` te muesta por cada capa: qué produjo, forma entrada→salida, dtype, min/max/media/NaN, tiempo, la métrica propia de la capa (tasa de detección de MediaPipe, % interpolado, accuracy del batch, WER del batch, BLEU) y una muestra de los valores que **entrega a la capa siguiente**.


## Limitacion con respecto al uso
La salida es la **secuencia léxica de glosas** y un inglés acotado al dominio hotelero — no prosa libre.
