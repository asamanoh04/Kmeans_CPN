# ============================================================
# generar_speedups.py
# Genera CSVs con los speedups para 2D y 3D
# Uso: python generar_speedups.py
# Debe correrse desde la carpeta raiz del proyecto (Kmeans_CPN/)
# ============================================================

import pandas as pd
import os

# ============================================================
# CONFIGURACION
# ============================================================
CSV_ENTRADA  = "./resultados/resultados_experimentos.csv"
CARPETA_SALIDA = "./resultados/speed_ups"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

HILOS    = [1, 6, 12, 24]
TAMANIOS = [100000, 200000, 300000, 400000, 600000, 800000, 1000000]

# ============================================================
# CARGAR DATOS
# ============================================================
df = pd.read_csv(CSV_ENTRADA)
serial   = df[df["tipo"] == "serial"].copy()
paralelo = df[df["tipo"] == "paralelo"].copy()

# ============================================================
# GENERAR CSV DE SPEEDUPS POR DIMENSION
# ============================================================
for dim in ["2D", "3D"]:
    filas = []

    for puntos in TAMANIOS:
        # Tiempo serial para este tamaño
        t_serial = serial[
            (serial["dimension"] == dim) &
            (serial["puntos"] == puntos)
        ]["promedio"].values

        if len(t_serial) == 0:
            continue

        for hilos_val in HILOS:
            t_paralelo = paralelo[
                (paralelo["dimension"] == dim) &
                (paralelo["puntos"] == puntos) &
                (paralelo["hilos"] == hilos_val)
            ]["promedio"].values

            if len(t_paralelo) == 0:
                continue

            speedup = round(t_serial[0] / t_paralelo[0], 4)

            filas.append({
                "dimension":       dim,
                "puntos":          puntos,
                "hilos":           hilos_val,
                "tiempo_serial":   round(t_serial[0], 6),
                "tiempo_paralelo": round(t_paralelo[0], 6),
                "speedup":         speedup
            })

    resultado = pd.DataFrame(filas)
    nombre = f"{CARPETA_SALIDA}/speedups_{dim.lower()}.csv"
    resultado.to_csv(nombre, index=False)
    print(f"Guardado: {nombre}")
    print(resultado.to_string(index=False))
    print()

print("Listo!")