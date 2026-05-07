// K-MEANS SERIAL - 3D
// Computo Paralelo - ITAM 2026
//
// Version base del K-means para datos en 3 dimensiones (x, y, z).
// Sin paralelizacion, todo corre en un solo hilo de forma secuencial.
// Es practicamente identico al serial 2D, solo que cada punto
// ahora tiene una coordenada z adicional.
//
// Como se corre:
// ./kmeans_serial_3d ../datos/100000_data_3d.csv 3 ../resultados/100000_salida_3d.csv
//   archivo de entrada    K centroides    donde guardar los resultados

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <chrono>

// Un punto en 3D con sus tres coordenadas y su cluster asignado.
// La unica diferencia vs 2D es que ahora tiene z tambien.
struct Punto3D {
    double x, y, z;
    int cluster;
};

// Un centroide en 3D, solo coordenadas.
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
// Se puede abrir en Python para graficar los clusters en 3D.
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

// Distancia euclidiana al cuadrado en 3D: dx^2 + dy^2 + dz^2.
// Sin raiz cuadrada porque solo comparamos distancias entre si,
// no necesitamos el valor exacto.
double distancia2(const Punto3D& p, const Centroide3D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    double dz = p.z - c.z;
    return dx*dx + dy*dy + dz*dz;
}

// Elige K puntos al azar del dataset como centroides de arranque.
// srand(42) para reproducibilidad: siempre los mismos centroides iniciales.
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

// El algoritmo K-means serial en 3D. Todo secuencial, punto por punto.
// Alterna entre asignar puntos y mover centroides hasta que nada cambie
// o se llegue al limite de 100 iteraciones.
// Regresa cuantas iteraciones tomo converger.
int kmeans(std::vector<Punto3D>& puntos, std::vector<Centroide3D>& centroides, int K, int max_iter = 100) {
    int n = puntos.size();
    int iter = 0;
    bool cambio = true;

    while (cambio && iter < max_iter) {
        cambio = false;
        iter++;

        // PASO 1: cada punto se va con el centroide mas cercano.
        // Si algun punto cambia de cluster, seguimos iterando.
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

        // PASO 2: cada centroide se mueve al promedio de sus puntos.
        // Ahora acumulamos sumas para x, y, z por separado.
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<double> suma_z(K, 0.0);
        std::vector<int> conteo(K, 0);

        for (int i = 0; i < n; i++) {
            int k = puntos[i].cluster;
            suma_x[k] += puntos[i].x;
            suma_y[k] += puntos[i].y;
            suma_z[k] += puntos[i].z;
            conteo[k]++;
        }

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

// Punto de entrada. Recibe 3 argumentos:
// 1. archivo CSV de entrada (x,y,z)
// 2. K numero de clusters
// 3. archivo CSV de salida
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