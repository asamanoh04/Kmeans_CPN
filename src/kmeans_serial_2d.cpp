// K-MEANS SERIAL - 2D
// Computo Paralelo - ITAM 2026
//
// Esta es la version base del K-means, sin paralelizacion.
// Sirve como punto de comparacion para medir el speedup
// que obtenemos con la version paralela.
//
// Como se corre:
// ./kmeans_serial_2d ../datos/100000_data_2d.csv 3 ../resultados/100000_salida_2d.csv
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

// Un punto en 2D con sus coordenadas y el cluster al que pertenece.
// cluster empieza en -1 porque al inicio ningun punto tiene cluster asignado.
struct Punto2D {
    double x, y;
    int cluster;
};

// Un centroide en 2D, solo necesita coordenadas.
struct Centroide2D {
    double x, y;
};

// Lee el CSV de entrada linea por linea.
// Espera dos valores por fila separados por coma: x, y.
// Sin encabezado, directo los numeros.
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

// Guarda los resultados en un CSV con tres columnas: x, y, cluster.
// Ese archivo se puede abrir en Python para visualizar los clusters con colores.
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

// Calcula la distancia euclidiana al cuadrado entre un punto y un centroide.
// No usamos raiz cuadrada porque solo necesitamos comparar distancias entre si,
// no el valor exacto. Esto nos ahorra millones de operaciones costosas.
double distancia2(const Punto2D& p, const Centroide2D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx*dx + dy*dy;
}

// Elige K puntos al azar del dataset como centroides iniciales.
// srand(42) fija la semilla para que siempre arranquen en el mismo lugar
// y los resultados sean reproducibles entre corridas.
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

// El algoritmo K-means serial. Todo en un solo hilo, punto por punto.
// Repite dos pasos hasta converger o llegar al maximo de iteraciones:
//   1. Asignar cada punto al centroide mas cercano
//   2. Mover cada centroide al promedio de sus puntos
// Regresa cuantas iteraciones tomo converger.
int kmeans(std::vector<Punto2D>& puntos, std::vector<Centroide2D>& centroides, int K, int max_iter = 100) {
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
        // Sumamos todas las coordenadas de cada cluster y dividimos entre cuantos son.
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

// Punto de entrada. Recibe 3 argumentos:
// 1. archivo CSV de entrada
// 2. K numero de clusters
// 3. archivo CSV de salida
int main(int argc, char* argv[]) {

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

    std::cout << "Leyendo datos de: " << archivo_entrada << std::endl;
    std::vector<Punto2D> puntos = leerCSV(archivo_entrada);
    std::cout << "Puntos leidos: " << puntos.size() << std::endl;
    std::cout << "K (clusters): " << K << std::endl;

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