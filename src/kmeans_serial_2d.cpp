// ============================================================
// K-MEANS SERIAL - 2D
// Computo Paralelo - ITAM 2026
// Uso: ./kmeans_serial_2d <archivo_entrada.csv> <K> <archivo_salida.csv>
// Ejemplo: ./kmeans_serial_2d ../datos/100000_data_2d.csv 3 ../resultados/100000_salida_2d.csv
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <chrono>

// ============================================================
// ESTRUCTURA: Un punto en 2D
// ============================================================
struct Punto2D {
    double x, y;
    int cluster; // A qué cluster pertenece este punto
};

// ============================================================
// ESTRUCTURA: Un centroide en 2D
// ============================================================
struct Centroide2D {
    double x, y;
};

// ============================================================
// FUNCIÓN: Leer CSV de entrada
// Formato esperado: x,y (sin encabezado)
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
// FUNCIÓN: Guardar CSV de salida
// Formato: x,y,cluster
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
// FUNCIÓN: Distancia euclidiana al cuadrado entre punto y centroide
// (No usamos raíz cuadrada para ahorrar cómputo — solo necesitamos comparar distancias)
// ============================================================
double distancia2(const Punto2D& p, const Centroide2D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx*dx + dy*dy;
}

// ============================================================
// FUNCIÓN: Inicializar centroides aleatoriamente
// Toma K puntos aleatorios del dataset como centroides iniciales
// ============================================================
std::vector<Centroide2D> inicializarCentroides(const std::vector<Punto2D>& puntos, int K) {
    std::vector<Centroide2D> centroides(K);
    int n = puntos.size();

    // Semilla aleatoria
    srand(42); // Fija la semilla para reproducibilidad

    for (int k = 0; k < K; k++) {
        int idx = rand() % n;
        centroides[k].x = puntos[idx].x;
        centroides[k].y = puntos[idx].y;
    }

    return centroides;
}

// ============================================================
// FUNCIÓN PRINCIPAL: Algoritmo K-means SERIAL
// Devuelve el número de iteraciones que tomó converger
// ============================================================
int kmeans(std::vector<Punto2D>& puntos, std::vector<Centroide2D>& centroides, int K, int max_iter = 100) {
    int n = puntos.size();
    int iter = 0;
    bool cambio = true;

    while (cambio && iter < max_iter) {
        cambio = false;
        iter++;

        // ----------------------------------------
        // PASO 1: Asignar cada punto al centroide más cercano
        // ----------------------------------------
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

            // Si el cluster del punto cambió, registrar que hubo cambio
            if (puntos[i].cluster != mejor_cluster) {
                puntos[i].cluster = mejor_cluster;
                cambio = true;
            }
        }

        // ----------------------------------------
        // PASO 2: Recalcular posición de cada centroide
        // Promedio de todos los puntos que le pertenecen
        // ----------------------------------------
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<int> conteo(K, 0);

        for (int i = 0; i < n; i++) {
            int k = puntos[i].cluster;
            suma_x[k] += puntos[i].x;
            suma_y[k] += puntos[i].y;
            conteo[k]++;
        }

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

    // Validar argumentos
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <entrada.csv> <K> <salida.csv>" << std::endl;
        std::cerr << "Ejemplo: ./kmeans_serial_2d ../datos/100000_data_2d.csv 3 ../resultados/100000_salida_2d.csv" << std::endl;
        return 1;
    }

    std::string archivo_entrada = argv[1];
    int K = std::atoi(argv[2]);
    std::string archivo_salida = argv[3];

    if (K <= 0) {
        std::cerr << "ERROR: K debe ser mayor que 0" << std::endl;
        return 1;
    }

    // Leer datos
    std::cout << "Leyendo datos de: " << archivo_entrada << std::endl;
    std::vector<Punto2D> puntos = leerCSV(archivo_entrada);
    std::cout << "Puntos leídos: " << puntos.size() << std::endl;
    std::cout << "K (clusters): " << K << std::endl;

    // Inicializar centroides
    std::vector<Centroide2D> centroides = inicializarCentroides(puntos, K);

    // Medir tiempo de ejecución
    auto inicio = std::chrono::high_resolution_clock::now();

    // Ejecutar K-means
    int iteraciones = kmeans(puntos, centroides, K);

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo = fin - inicio;

    // Mostrar resultados
    std::cout << "Iteraciones: " << iteraciones << std::endl;
    std::cout << "Tiempo (segundos): " << tiempo.count() << std::endl;

    // Guardar resultados
    guardarCSV(archivo_salida, puntos);
    std::cout << "Resultados guardados en: " << archivo_salida << std::endl;

    return 0;
}