// ============================================================
// K-MEANS PARALELO - 2D (OpenMP)
// Computo Paralelo - ITAM 2026
// Uso: ./kmeans_paralelo_2d <entrada.csv> <K> <salida.csv> <num_hilos>
// Ejemplo: ./kmeans_paralelo_2d ../datos/100000_data_2d.csv 3 ../resultados/salida.csv 6
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <omp.h>       // ← única librería nueva vs serial

// ============================================================
// ESTRUCTURAS (idénticas al serial)
// ============================================================
struct Punto2D {
    double x, y;
    int cluster;
};

struct Centroide2D {
    double x, y;
};

// ============================================================
// FUNCIÓN: Leer CSV (idéntica al serial)
// ============================================================
std::vector<Punto2D> leerCSV(const std::string& archivo) {
    std::vector<Punto2D> puntos;
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
        Punto2D p;
        p.cluster = -1;

        std::getline(ss, val, ','); p.x = std::stod(val);
        std::getline(ss, val, ','); p.y = std::stod(val);

        puntos.push_back(p);
    }

    f.close();
    return puntos;
}

// ============================================================
// FUNCIÓN: Guardar CSV (idéntica al serial)
// ============================================================
void guardarCSV(const std::string& archivo, const std::vector<Punto2D>& puntos) {
    std::ofstream f(archivo);

    if (!f.is_open()) {
        std::cerr << "ERROR: No se pudo crear el archivo: " << archivo << std::endl;
        exit(1);
    }

    f << "x,y,cluster\n";
    for (const auto& p : puntos) {
        f << p.x << "," << p.y << "," << p.cluster << "\n";
    }

    f.close();
}

// ============================================================
// FUNCIÓN: Distancia euclidiana al cuadrado (idéntica al serial)
// ============================================================
double distancia2(const Punto2D& p, const Centroide2D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx*dx + dy*dy;
}

// ============================================================
// FUNCIÓN: Inicializar centroides (idéntica al serial)
// ============================================================
std::vector<Centroide2D> inicializarCentroides(const std::vector<Punto2D>& puntos, int K) {
    std::vector<Centroide2D> centroides(K);
    int n = puntos.size();
    srand(42);
    for (int k = 0; k < K; k++) {
        int idx = rand() % n;
        centroides[k].x = puntos[idx].x;
        centroides[k].y = puntos[idx].y;
    }
    return centroides;
}

// ============================================================
// FUNCIÓN PRINCIPAL: K-means PARALELO con OpenMP
// ← AQUÍ están los cambios vs serial
// ============================================================
int kmeans(std::vector<Punto2D>& puntos, std::vector<Centroide2D>& centroides, int K, int max_iter = 100) {
    int n = puntos.size();
    int iter = 0;
    bool cambio = true;

    while (cambio && iter < max_iter) {
        cambio = false;
        iter++;

        // ----------------------------------------
        // PASO 1: Asignar puntos a centroides - PARALELIZADO
        // Cada hilo procesa un subconjunto de puntos
        // ----------------------------------------
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

        // ----------------------------------------
        // PASO 2: Recalcular centroides - PARALELIZADO
        // Cada hilo acumula sumas parciales y al final se combinan
        // ----------------------------------------
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<int> conteo(K, 0);

        #pragma omp parallel
        {
            // Cada hilo tiene sus propias sumas locales para evitar conflictos
            std::vector<double> local_x(K, 0.0);
            std::vector<double> local_y(K, 0.0);
            std::vector<int> local_conteo(K, 0);

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                int k = puntos[i].cluster;
                local_x[k] += puntos[i].x;
                local_y[k] += puntos[i].y;
                local_conteo[k]++;
            }

            // Combinar sumas locales en las globales (seccion critica)
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    suma_x[k] += local_x[k];
                    suma_y[k] += local_y[k];
                    conteo[k] += local_conteo[k];
                }
            }
        }

        // Actualizar posición de centroides
        for (int k = 0; k < K; k++) {
            if (conteo[k] > 0) {
                centroides[k].x = suma_x[k] / conteo[k];
                centroides[k].y = suma_y[k] / conteo[k];
            }
        }
    }

    return iter;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Uso: " << argv[0] << " <entrada.csv> <K> <salida.csv> <num_hilos>" << std::endl;
        std::cerr << "Ejemplo: ./kmeans_paralelo_2d ../datos/100000_data_2d.csv 3 ../resultados/salida.csv 6" << std::endl;
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
    std::vector<Punto2D> puntos = leerCSV(archivo_entrada);
    std::cout << "Puntos leidos: " << puntos.size() << std::endl;
    std::cout << "K (clusters): " << K << std::endl;
    std::cout << "Hilos: " << num_hilos << std::endl;

    std::vector<Centroide2D> centroides = inicializarCentroides(puntos, K);

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