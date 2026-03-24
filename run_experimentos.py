# ============================================================
# run_experimentos.py
# Corre todos los experimentos de K-means y guarda los tiempos
# Computo Paralelo - ITAM 2026
#
# Uso: python run_experimentos.py
# Debe correrse desde la carpeta raiz del proyecto (Kmeans_CPN/)
# ============================================================

import subprocess
import os
import csv

# ============================================================
# PARAMETROS DEL EXPERIMENTO
# ============================================================
K            = 3
REPETICIONES = 10
HILOS        = [1, 6, 12, 24]
TAMANIOS     = [100000, 200000, 300000, 400000, 600000, 800000, 1000000]
DIMENSIONES  = ["2d", "3d"]

# Rutas
SRC          = "./src"
DATOS        = "./datos"
RESULTADOS   = "./resultados"
SALIDA_CSV   = "./resultados/resultados_experimentos.csv"

# Ejecutables
SERIAL   = {"2d": f"{SRC}/kmeans_serial_2d",   "3d": f"{SRC}/kmeans_serial_3d"}
PARALELO = {"2d": f"{SRC}/kmeans_paralelo_2d", "3d": f"{SRC}/kmeans_paralelo_3d"}

# ============================================================
# FUNCIÓN: Corre un ejecutable y regresa el tiempo en segundos
# ============================================================
def correr_kmeans(comando):
    try:
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=300
        )

        for linea in resultado.stdout.split("\n"):
            if "Tiempo (segundos)" in linea:
                tiempo = float(linea.split(":")[1].strip())
                return tiempo

        print(f"ERROR en comando: {' '.join(comando)}")
        print(resultado.stderr)
        return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT en: {' '.join(comando)}")
        return None
    except Exception as e:
        print(f"EXCEPCION: {e}")
        return None

# ============================================================
# FUNCIÓN: Corre un experimento N veces y regresa todos los tiempos
# ============================================================
def correr_experimento(comando, repeticiones):
    tiempos = []
    for rep in range(repeticiones):
        t = correr_kmeans(comando)
        if t is not None:
            tiempos.append(t)
    return tiempos

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTADOS, exist_ok=True)

    with open(SALIDA_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        # Encabezado
        encabezado = ["dimension", "puntos", "tipo", "hilos"] + \
                     [f"t{i+1}" for i in range(REPETICIONES)] + \
                     ["promedio"]
        writer.writerow(encabezado)

        total_experimentos = len(DIMENSIONES) * len(TAMANIOS) * (1 + len(HILOS))
        experimento_actual = 0

        for dim in DIMENSIONES:
            for puntos in TAMANIOS:

                archivo_entrada = f"{DATOS}/{puntos}_data_{dim}.csv"
                archivo_salida  = f"{RESULTADOS}/temp_salida_{dim}.csv"

                # ----------------------------------------
                # SERIAL
                # ----------------------------------------
                experimento_actual += 1
                print(f"[{experimento_actual}/{total_experimentos}] Serial {dim.upper()} - {puntos:,} puntos...")

                comando_serial = [
                    SERIAL[dim],
                    archivo_entrada,
                    str(K),
                    archivo_salida
                ]

                tiempos = correr_experimento(comando_serial, REPETICIONES)

                if tiempos:
                    promedio = sum(tiempos) / len(tiempos)
                    fila = [dim.upper(), puntos, "serial", 1] + tiempos + [round(promedio, 6)]
                    writer.writerow(fila)
                    f.flush()
                    print(f"   Promedio: {promedio:.4f}s")

                # ----------------------------------------
                # PARALELO con diferentes hilos
                # ----------------------------------------
                for hilos in HILOS:
                    experimento_actual += 1
                    print(f"[{experimento_actual}/{total_experimentos}] Paralelo {dim.upper()} - {puntos:,} puntos - {hilos} hilos...")

                    comando_paralelo = [
                        PARALELO[dim],
                        archivo_entrada,
                        str(K),
                        archivo_salida,
                        str(hilos)
                    ]

                    tiempos = correr_experimento(comando_paralelo, REPETICIONES)

                    if tiempos:
                        promedio = sum(tiempos) / len(tiempos)
                        fila = [dim.upper(), puntos, "paralelo", hilos] + tiempos + [round(promedio, 6)]
                        writer.writerow(fila)
                        f.flush()
                        print(f"   Promedio: {promedio:.4f}s")

    print(f"\nExperimentos terminados.")
    print(f"Resultados guardados en: {SALIDA_CSV}")

if __name__ == "__main__":
    main()