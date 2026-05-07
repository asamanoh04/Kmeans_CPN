"""
Script para generar datos sintéticos para K-means
Genera archivos CSV con puntos aleatorios en 2D y 3D
Parámetros configurables: K (clusters) y tamaños de datos
"""
 
from sklearn.datasets import make_blobs
import numpy as np
 
# ============================
# PARÁMETROS (CAMBIA AQUÍ)
# ============================
K = 3  # Número de clusters (centroides)
SIZES = [100000, 200000, 300000, 400000, 600000, 800000, 1000000]
 
# ============================
# GENERAR DATOS 2D
# ============================
print("Generando datos 2D...")
for size in SIZES:
    print(f"  Generando {size} puntos 2D...", end=" ")
    
    # Generar puntos
    points, _ = make_blobs(
        n_samples=size,
        centers=K,
        n_features=2,
        cluster_std=0.04,
        random_state=7,
        center_box=(0, 1.0)
    )
    
    # Solo valores positivos, 3 decimales
    points = np.round(np.abs(points), 3)
    
    # Guardar CSV
    filename = f"{size}_data_2d.csv"
    np.savetxt(filename, points, delimiter=",", fmt="%.3f")
    print(f"✓ Guardado: {filename}")
 
# ============================
# GENERAR DATOS 3D
# ============================
print("\nGenerando datos 3D...")
for size in SIZES:
    print(f"  Generando {size} puntos 3D...", end=" ")
    
    # Generar puntos
    points, _ = make_blobs(
        n_samples=size,
        centers=K,
        n_features=3,
        cluster_std=0.04,
        random_state=7,
        center_box=(0, 1.0)
    )
    
    # Solo valores positivos, 3 decimales
    points = np.round(np.abs(points), 3)
    
    # Guardar CSV
    filename = f"{size}_data_3d.csv"
    np.savetxt(filename, points, delimiter=",", fmt="%.3f")
    print(f"✓ Guardado: {filename}")
 
print("\n✓ ¡LISTO! Todos los archivos generados.")
print(f"Total: {len(SIZES) * 2} archivos CSV")