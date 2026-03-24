// ============================================================
// K-MEANS SERIAL - 3D
// Computo Paralelo - ITAM 2026
// Uso: ./kmeans_serial_3d <archivo_entrada.csv> <K> <archivo_salida.csv>
// Ejemplo: ./kmeans_serial_3d ../datos/100000_data_3d.csv 3 ../resultados/100000_salida_3d.csv
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
// ESTRUCTURA: Un punto en 3D
// ============================================================
struct Punto3D {
    double x, y, z;  // ← única diferencia vs 2D: agrega z
    int cluster;
};

// ============================================================
// ESTRUCTURA: Un centroide en 3D
// ============================================================
struct Centroide3D {
    double x, y, z;  // ← única diferencia vs 2D: agrega z
};

// ============================================================
// FUNCIÓN: Leer CSV de entrada
// Formato esperado: x,y,z (sin encabezado)
// ============================================================
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
        std::getline(ss, val, ','); p.z = std::stod(val);  // ← lee z también

        puntos.push_back(p);
    }

    f.close();
    return puntos;
}

// ============================================================
// FUNCIÓN: Guardar CSV de salida
// Formato: x,y,z,cluster
// ============================================================
void guardarCSV(const std::string& archivo, const std::vector<Punto3D>& puntos) {
    std::ofstream f(archivo);

    if (!f.is_open()) {
        std::cerr << "ERROR: No se pudo crear el archivo: " << archivo << std::endl;
        exit(1);
    }

    f << "x,y,z,cluster\n";  // ← encabezado con z
    for (const auto& p : puntos) {
        f << p.x << "," << p.y << "," << p.z << "," << p.cluster << "\n";
    }

    f.close();
}

// ============================================================
// FUNCIÓN: Distancia euclidiana al cuadrado entre punto y centroide
// En 3D: dx² + dy² + dz²
// ============================================================
double distancia2(const Punto3D& p, const Centroide3D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    double dz = p.z - c.z;  // ← agrega dz
    return dx*dx + dy*dy + dz*dz;
}

// ============================================================
// FUNCIÓN: Inicializar centroides aleatoriamente
// ============================================================
std::vector<Centroide3D> inicializarCentroides(const std::vector<Punto3D>& puntos, int K) {
    std::vector<Centroide3D> centroides(K);
    int n = puntos.size();

    srand(42);

    for (int k = 0; k < K; k++) {
        int idx = rand() % n;
        centroides[k].x = puntos[idx].x;
        centroides[k].y = puntos[idx].y;
        centroides[k].z = puntos[idx].z;  // ← inicializa z también
    }

    return centroides;
}

// ============================================================
// FUNCIÓN PRINCIPAL: Algoritmo K-means SERIAL 3D
// ============================================================
int kmeans(std::vector<Punto3D>& puntos, std::vector<Centroide3D>& centroides, int K, int max_iter = 100) {
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

            if (puntos[i].cluster != mejor_cluster) {
                puntos[i].cluster = mejor_cluster;
                cambio = true;
            }
        }

        // ----------------------------------------
        // PASO 2: Recalcular posición de cada centroide
        // ----------------------------------------
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<double> suma_z(K, 0.0);  // ← agrega suma_z
        std::vector<int> conteo(K, 0);

        for (int i = 0; i < n; i++) {
            int k = puntos[i].cluster;
            suma_x[k] += puntos[i].x;
            suma_y[k] += puntos[i].y;
            suma_z[k] += puntos[i].z;  // ← acumula z
            conteo[k]++;
        }

        for (int k = 0; k < K; k++) {
            if (conteo[k] > 0) {
                centroides[k].x = suma_x[k] / conteo[k];
                centroides[k].y = suma_y[k] / conteo[k];
                centroides[k].z = suma_z[k] / conteo[k];  // ← actualiza z
            }
        }
    }

    return iter;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <entrada.csv> <K> <salida.csv>" << std::endl;
        std::cerr << "Ejemplo: ./kmeans_serial_3d ../datos/100000_data_3d.csv 3 ../resultados/100000_salida_3d.csv" << std::endl;
        return 1;
    }

    std::string archivo_entrada = argv[1];
    int K = std::atoi(argv[2]);
    std::string archivo_salida = argv[3];

    if (K <= 0) {
        std::cerr << "ERROR: K debe ser mayor que 0" << std::endl;
        return 1;
    }

    std::cout << "Leyendo datos de: " << archivo_entrada << std::endl;
    std::vector<Punto3D> puntos = leerCSV(archivo_entrada);
    std::cout << "Puntos leidos: " << puntos.size() << std::endl;
    std::cout << "K (clusters): " << K << std::endl;

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