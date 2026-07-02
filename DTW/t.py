import pandas as pd

# Prueba diferentes delimitadores
for delim in ['\t', ',', ';']:
    try:
        df = pd.read_csv("datos_30.csv", delimiter=delim, nrows=5)
        print(f"\nDelimitador: {repr(delim)}")
        print(f"Columnas: {list(df.columns)}")
        if 'glosa' in df.columns:
            print(f"Glosas en primeras 5 filas: {df['glosa'].tolist()}")
    except:
        pass

# Cargar completo con el delimitador correcto
df = pd.read_csv("datos_30.csv", delimiter='\t')
print(f"\n=== ESTADÍSTICAS ===")
print(f"Filas totales: {len(df)}")
print(f"Videos únicos: {df['video'].nunique()}")
print(f"Glosas únicas: {df['glosa'].nunique()}")
print(f"Valores en 'video': {df['video'].unique()[:10]}")
print(f"Valores en 'glosa': {df['glosa'].unique()}")
print(f"\nConteo por glosa:")
print(df['glosa'].value_counts())