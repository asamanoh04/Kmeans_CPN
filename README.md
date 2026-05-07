# K-means Paralelo con OpenMP
**Computo Paralelo y En La Nube — ITAM 2026**  
Andres Samano · Alejandro Salas

---

## ¿De qué va esto?

Implementación del algoritmo de clustering K-means en C++, con una versión serial y una versión paralelizada usando OpenMP. El objetivo es medir qué tanto speedup se puede obtener al distribuir el trabajo entre múltiples hilos en una misma máquina.

Se implementaron versiones para datos en **2D (x, y)** y **3D (x, y, z)**, probando con datasets de hasta **1 millón de puntos** y diferentes configuraciones de hilos.

---

## Estructura del proyecto

```
Kmeans_CPN/
├── src/
│   ├── kmeans_serial_2d.cpp      # K-means serial para datos 2D
│   ├── kmeans_serial_3d.cpp      # K-means serial para datos 3D
│   ├── kmeans_paralelo_2d.cpp    # K-means paralelo (OpenMP) para datos 2D
│   └── kmeans_paralelo_3d.cpp    # K-means paralelo (OpenMP) para datos 3D
├── documentos/
│   ├── reporte_kmeans.pdf        # Reporte completo del proyecto
│   └── reporte_kmeans.docx       # Reporte en Word
├── generar_datos.py              # Genera los CSVs de entrada
├── run_experimentos.py           # Corre todos los experimentos automaticamente
├── graficas_resultados.py        # Genera las graficas de speedup
└── generar_speedups.py           # Calcula y guarda los speedups en CSV
```

---

## Cómo correrlo desde cero

### 1. Clonar el repo
```bash
git clone https://github.com/asamanoh04/Kmeans_CPN.git
cd Kmeans_CPN
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
.\venv\Scripts\activate
pip install scikit-learn numpy pandas seaborn matplotlib
```

### 3. Generar los datos de entrada
```bash
python generar_datos.py
```
Esto crea 14 archivos CSV en `datos/` (7 tamaños × 2D y 3D).

### 4. Compilar los ejecutables
```bash
cd src
g++ -O2 -o kmeans_serial_2d kmeans_serial_2d.cpp
g++ -O2 -o kmeans_serial_3d kmeans_serial_3d.cpp
g++ -O2 -fopenmp -o kmeans_paralelo_2d kmeans_paralelo_2d.cpp
g++ -O2 -fopenmp -o kmeans_paralelo_3d kmeans_paralelo_3d.cpp
cd ..
```

### 5. Correr los experimentos
```bash
python run_experimentos.py
```
Corre todos los ejecutables con todos los tamaños de datos y configuraciones de hilos. Tarda unos minutos. Los tiempos se guardan en `resultados/resultados_experimentos.csv`.

### 6. Generar gráficas y speedups
```bash
python graficas_resultados.py
python generar_speedups.py
```

---

## Parámetros del experimento

| Parámetro | Valores |
|---|---|
| Dimensiones | 2D y 3D |
| Número de puntos | 100K, 200K, 300K, 400K, 600K, 800K, 1M |
| Clusters (K) | 3 |
| Hilos | 1, 6, 12, 24 |
| Repeticiones | 10 por configuración (se promedia) |

---

## Resultados destacados

- Mejor speedup en **2D**: **1.66x** con 12 hilos y 1M puntos
- Mejor speedup en **3D**: **2.43x** con 12 hilos y 1M puntos
- Para datasets pequeños (< 400K puntos) el overhead de hilos supera la ganancia

---

## Hardware usado

- **Laptop**: Razer Blade 15 Base Early 2021
- **Procesador**: Intel Core i7-10750H @ 2.60GHz
- **Cores**: 6 físicos / 12 lógicos (Hyper-Threading)
- **RAM**: 16 GB
- **SO**: Windows 10 Home
- **Compilador**: g++ MinGW con `-O2 -fopenmp`
