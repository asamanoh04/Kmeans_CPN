// K-MEANS PARALELO - 3D con OpenMP
// Computo Paralelo - ITAM 2026
//
// Version paralela del K-means para datos en 3 dimensiones (x, y, z).
// Identica en logica al paralelo 2D, solo que ahora cada punto
// tiene una coordenada z adicional que hay que considerar en todo.
//
// Como se corre:
// ./kmeans_paralelo_3d ../datos/100000_data_3d.csv 3 ../resultados/salida.csv 6
//   archivo de entrada    K centroides    donde guardar    cuantos hilos usar

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <omp.h>

// Un punto en 3D con sus tres coordenadas y su cluster asignado.
// cluster empieza en -1 (sin asignar).
struct Punto3D {
    double x, y, z;
    int cluster;
};

// Un centroide en 3D, solo necesita coordenadas.
struct Centroide3D {
    double x, y, z;
};

// Lee el CSV linea por linea esperando tres valores por fila: x, y, z.
// Sin encabezado, directo los numeros desde la primera linea.
std::vector<Punto3D> leerCSV(const std::string& archivo) {
    std::vector<Punto3D> puntos;
    std::ifstream f(archivo);

    if (!f.is_open()) {
        std::cerr << "ERROR: No se pudo abrir el archivo: " << archivo << std::endl;
        exit(1);
    }

    std::string linea;
    while (std::getline(f, linea)) {
        if (linea.empty()) continue;
        std::stringstream ss(linea);
        std::string val;
        Punto3D p;
        p.cluster = -1;

        std::getline(ss, val, ','); p.x = std::stod(val);
        std::getline(ss, val, ','); p.y = std::stod(val);
        std::getline(ss, val, ','); p.z = std::stod(val);

        puntos.push_back(p);
    }

    f.close();
    return puntos;
}

// Guarda los resultados en CSV con cuatro columnas: x, y, z, cluster.
// Este archivo se puede abrir en Python para visualizar los clusters en 3D.
void guardarCSV(const std::string& archivo, const std::vector<Punto3D>& puntos) {
    std::ofstream f(archivo);

    if (!f.is_open()) {
        std::cerr << "ERROR: No se pudo crear el archivo: " << archivo << std::endl;
        exit(1);
    }

    f << "x,y,z,cluster\n";
    for (const auto& p : puntos) {
        f << p.x << "," << p.y << "," << p.z << "," << p.cluster << "\n";
    }

    f.close();
}

// Distancia euclidiana al cuadrado en 3D.
// Igual que en 2D pero ahora sumamos tambien la diferencia en z.
// Sin raiz cuadrada porque solo comparamos distancias, no necesitamos el valor exacto.
double distancia2(const Punto3D& p, const Centroide3D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    double dz = p.z - c.z;
    return dx*dx + dy*dy + dz*dz;
}

// Elige K puntos al azar del dataset como centroides de arranque.
// srand(42) fija la semilla para que el experimento sea reproducible.
std::vector<Centroide3D> inicializarCentroides(const std::vector<Punto3D>& puntos, int K) {
    std::vector<Centroide3D> centroides(K);
    int n = puntos.size();
    srand(42);
    for (int k = 0; k < K; k++) {
        int idx = rand() % n;
        centroides[k].x = puntos[idx].x;
        centroides[k].y = puntos[idx].y;
        centroides[k].z = puntos[idx].z;
    }
    return centroides;
}

// El algoritmo K-means en 3D con OpenMP.
// Misma estrategia que el 2D: paralelizar la asignacion de puntos
// y el recalculo de centroides con sumas locales por hilo.
// Regresa cuantas iteraciones tomo converger.
int kmeans(std::vector<Punto3D>& puntos, std::vector<Centroide3D>& centroides, int K, int max_iter = 100) {
    int n = puntos.size();
    int iter = 0;
    bool cambio = true;

    while (cambio && iter < max_iter) {
        cambio = false;
        iter++;

        // PASO 1: cada punto busca el centroide mas cercano.
        // Los hilos se reparten los puntos en bloques iguales (schedule static).
        // reduction(||:cambio) detecta si algun punto cambio de cluster
        // sin que los hilos se pisen entre si.
        #pragma omp parallel for schedule(static) shared(puntos, centroides) reduction(||:cambio)
        for (int i = 0; i < n; i++) {
            double mejor_dist = std::numeric_limits<double>::max();
            int mejor_cluster = 0;

            for (int k = 0; k < K; k++) {
                double d = distancia2(puntos[i], centroides[k]);
                if (d < mejor_dist) {
                    mejor_dist = d;
                    mejor_cluster = k;
                }
            }

            if (puntos[i].cluster != mejor_cluster) {
                puntos[i].cluster = mejor_cluster;
                cambio = true;
            }
        }

        // PASO 2: recalcular la posicion de cada centroide.
        // Cada hilo acumula sus propias sumas locales para x, y, z
        // y al final las junta en la seccion critica.
        // Asi evitamos condiciones de carrera sin sacrificar rendimiento.
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<double> suma_z(K, 0.0);
        std::vector<int> conteo(K, 0);

        #pragma omp parallel
        {
            // cada hilo tiene su propia copia de las sumas, nadie interfiere
            std::vector<double> local_x(K, 0.0);
            std::vector<double> local_y(K, 0.0);
            std::vector<double> local_z(K, 0.0);
            std::vector<int> local_conteo(K, 0);

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                int k = puntos[i].cluster;
                local_x[k] += puntos[i].x;
                local_y[k] += puntos[i].y;
                local_z[k] += puntos[i].z;
                local_conteo[k]++;
            }

            // cada hilo aporta su parte al total, de uno en uno
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    suma_x[k] += local_x[k];
                    suma_y[k] += local_y[k];
                    suma_z[k] += local_z[k];
                    conteo[k] += local_conteo[k];
                }
            }
        }

        // con las sumas completas, movemos cada centroide a su nuevo lugar
        for (int k = 0; k < K; k++) {
            if (conteo[k] > 0) {
                centroides[k].x = suma_x[k] / conteo[k];
                centroides[k].y = suma_y[k] / conteo[k];
                centroides[k].z = suma_z[k] / conteo[k];
            }
        }
    }

    return iter;
}

// Punto de entrada. Recibe 4 argumentos:
// 1. archivo CSV de entrada (x,y,z)
// 2. K numero de clusters
// 3. archivo CSV de salida
// 4. numero de hilos OpenMP a usar
int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Uso: " << argv[0] << " <entrada.csv> <K> <salida.csv> <num_hilos>" << std::endl;
        std::cerr << "Ejemplo: ./kmeans_paralelo_3d ../datos/100000_data_3d.csv 3 ../resultados/salida.csv 6" << std::endl;
        return 1;
    }

    std::string archivo_entrada = argv[1];
    int K         = std::atoi(argv[2]);
    std::string archivo_salida = argv[3];
    int num_hilos = std::atoi(argv[4]);

    if (K <= 0 || num_hilos <= 0) {
        std::cerr << "ERROR: K y num_hilos deben ser mayores que 0" << std::endl;
        return 1;
    }

    omp_set_num_threads(num_hilos);

    std::cout << "Leyendo datos de: " << archivo_entrada << std::endl;
    std::vector<Punto3D> puntos = leerCSV(archivo_entrada);
    std::cout << "Puntos leidos: " << puntos.size() << std::endl;
    std::cout << "K (clusters): " << K << std::endl;
    std::cout << "Hilos: " << num_hilos << std::endl;

    std::vector<Centroide3D> centroides = inicializarCentroides(puntos, K);

    auto inicio = std::chrono::high_resolution_clock::now();
    int iteraciones = kmeans(puntos, centroides, K);
    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo = fin - inicio;

    std::cout << "Iteraciones: " << iteraciones << std::endl;
    std::cout << "Tiempo (segundos): " << tiempo.count() << std::endl;

    guardarCSV(archivo_salida, puntos);
    std::cout << "Resultados guardados en: " << archivo_salida << std::endl;

    return 0;
}