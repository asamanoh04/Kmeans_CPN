# ============================================================
# plot_results.py
# Genera graficas de speedup y tiempos para el reporte
# Computo Paralelo - ITAM 2026
#
# Uso: python plot_results.py
# Debe correrse desde la carpeta raiz del proyecto (Kmeans_CPN/)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURACION
# ============================================================
CSV_ENTRADA  = "./resultados/resultados_experimentos.csv"
CARPETA_IMGS = "./resultados/graficas"
os.makedirs(CARPETA_IMGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="colorblind")

HILOS        = [1, 6, 12, 24]
TAMANIOS     = [100000, 200000, 300000, 400000, 600000, 800000, 1000000]
TAMANIOS_REP = [100000, 400000, 800000, 1000000]

# ============================================================
# CARGAR DATOS
# ============================================================
df = pd.read_csv(CSV_ENTRADA)
serial   = df[df["tipo"] == "serial"].copy()
paralelo = df[df["tipo"] == "paralelo"].copy()

# ============================================================
# FUNCIÓN: Calcular speedup
# ============================================================
def calcular_speedup(dim, puntos, hilos_val):
    t_serial = serial[
        (serial["dimension"] == dim) &
        (serial["puntos"] == puntos)
    ]["promedio"].values

    t_paralelo = paralelo[
        (paralelo["dimension"] == dim) &
        (paralelo["puntos"] == puntos) &
        (paralelo["hilos"] == hilos_val)
    ]["promedio"].values

    if len(t_serial) > 0 and len(t_paralelo) > 0 and t_paralelo[0] > 0:
        return round(t_serial[0] / t_paralelo[0], 4)
    return None

# ============================================================
# GRAFICA 1: Speedup vs Numero de Puntos
# ============================================================
def grafica_speedup_vs_puntos():
    for dim in ["2D", "3D"]:
        filas = []
        for hilos_val in HILOS:
            for puntos in TAMANIOS:
                sp = calcular_speedup(dim, puntos, hilos_val)
                if sp is not None:
                    filas.append({
                        "Puntos (K)": puntos // 1000,
                        "Speedup":    sp,
                        "Hilos":      str(hilos_val)
                    })

        data = pd.DataFrame(filas)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x="Puntos (K)", y="Speedup",
                     hue="Hilos", marker="o", ax=ax, linewidth=2.5, markersize=8)
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1.2)
        ax.set_title(f"Speedup vs Numero de Puntos — {dim}  |  K=3", fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Puntos (miles)", fontsize=12)
        ax.set_ylabel("Speedup  (t_serial / t_paralelo)", fontsize=12)
        ax.legend(title="Hilos", fontsize=10)
        plt.tight_layout()
        nombre = f"{CARPETA_IMGS}/speedup_vs_puntos_{dim.lower()}.png"
        plt.savefig(nombre, dpi=150)
        plt.close()
        print(f"Guardada: {nombre}")

# ============================================================
# GRAFICA 2: Tiempo vs Numero de Puntos
# ============================================================
def grafica_tiempo_vs_puntos():
    for dim in ["2D", "3D"]:
        filas = []
        for puntos in TAMANIOS:
            t = serial[
                (serial["dimension"] == dim) &
                (serial["puntos"] == puntos)
            ]["promedio"].values
            if len(t) > 0:
                filas.append({"Puntos (K)": puntos // 1000, "Tiempo (s)": t[0], "Version": "Serial"})

        for hilos_val in HILOS:
            for puntos in TAMANIOS:
                t = paralelo[
                    (paralelo["dimension"] == dim) &
                    (paralelo["puntos"] == puntos) &
                    (paralelo["hilos"] == hilos_val)
                ]["promedio"].values
                if len(t) > 0:
                    filas.append({"Puntos (K)": puntos // 1000, "Tiempo (s)": t[0], "Version": f"Paralelo {hilos_val} hilos"})

        data = pd.DataFrame(filas)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x="Puntos (K)", y="Tiempo (s)",
                     hue="Version", marker="o", ax=ax, linewidth=2.5, markersize=8)
        ax.set_title(f"Tiempo vs Numero de Puntos — {dim}  |  K=3", fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Puntos (miles)", fontsize=12)
        ax.set_ylabel("Tiempo Promedio (segundos)", fontsize=12)
        ax.legend(title="Version", fontsize=10)
        plt.tight_layout()
        nombre = f"{CARPETA_IMGS}/tiempo_vs_puntos_{dim.lower()}.png"
        plt.savefig(nombre, dpi=150)
        plt.close()
        print(f"Guardada: {nombre}")

# ============================================================
# GRAFICA 3: Speedup vs Numero de Hilos
# ============================================================
def grafica_speedup_vs_hilos():
    for dim in ["2D", "3D"]:
        filas = []
        for puntos in TAMANIOS_REP:
            for hilos_val in HILOS:
                sp = calcular_speedup(dim, puntos, hilos_val)
                if sp is not None:
                    filas.append({
                        "Hilos":      hilos_val,
                        "Speedup":    sp,
                        "Puntos (K)": f"{puntos // 1000}K"
                    })

        data = pd.DataFrame(filas)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x="Hilos", y="Speedup",
                     hue="Puntos (K)", marker="o", ax=ax, linewidth=2.5, markersize=8)
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1.2)
        ax.set_title(f"Speedup vs Numero de Hilos — {dim}  |  K=3", fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Hilos", fontsize=12)
        ax.set_ylabel("Speedup  (t_serial / t_paralelo)", fontsize=12)
        ax.set_xticks(HILOS)
        ax.legend(title="Puntos", fontsize=10)
        plt.tight_layout()
        nombre = f"{CARPETA_IMGS}/speedup_vs_hilos_{dim.lower()}.png"
        plt.savefig(nombre, dpi=150)
        plt.close()
        print(f"Guardada: {nombre}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Generando graficas con seaborn...\n")
    grafica_speedup_vs_puntos()
    grafica_tiempo_vs_puntos()
    grafica_speedup_vs_hilos()
    print(f"\nTodas las graficas guardadas en: {CARPETA_IMGS}/")