// K-MEANS PARALELO - 2D con OpenMP
// Computo Paralelo - ITAM 2026
//
// Este archivo es la version paralela del K-means para datos en 2D.
// La idea es la misma que el serial pero aprovechando multiples hilos
// para hacer el trabajo mas rapido.
//
// Como se corre:
// ./kmeans_paralelo_2d ../datos/100000_data_2d.csv 3 ../resultados/salida.csv 6
//   archivo de entrada    K centroides    donde guardar    cuantos hilos usar

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <omp.h>       // esto es lo unico nuevo vs el serial, es la libreria de OpenMP

// Un punto en 2D. Guarda sus coordenadas y a que cluster pertenece.
// Empieza en -1 porque todavia no tiene cluster asignado.
struct Punto2D {
    double x, y;
    int cluster;
};

// Un centroide en 2D. Solo necesita coordenadas, sin cluster.
struct Centroide2D {
    double x, y;
};

// Lee el CSV de entrada linea por linea y regresa un vector con todos los puntos.
// Si el archivo no existe o no se puede abrir, truena con error.
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
// Ese archivo despues se puede abrir en Python para visualizar los clusters.
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
// Usamos el cuadrado (sin raiz cuadrada) porque solo nos importa cual es
// el mas cercano, no la distancia exacta. Esto ahorra millones de sqrt().
double distancia2(const Punto2D& p, const Centroide2D& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx*dx + dy*dy;
}

// Elige K puntos al azar del dataset como centroides iniciales.
// Usamos srand(42) para que siempre salgan los mismos centroides iniciales
// y los experimentos sean reproducibles.
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

// El algoritmo K-means propiamente dicho, ahora con OpenMP.
// Hay dos partes paralelizadas: asignar puntos y recalcular centroides.
// Regresa cuantas iteraciones tomo converger.
int kmeans(std::vector<Punto2D>& puntos, std::vector<Centroide2D>& centroides, int K, int max_iter = 100) {
    int n = puntos.size();
    int iter = 0;
    bool cambio = true;

    while (cambio && iter < max_iter) {
        cambio = false;
        iter++;

        // PASO 1: cada punto busca su centroide mas cercano.
        // Con OpenMP dividimos los puntos entre los hilos disponibles,
        // cada hilo agarra su pedazo y trabaja de forma independiente.
        // La parte de "reduction(||:cambio)" es para detectar convergencia
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

        // PASO 2: mover cada centroide al promedio de sus puntos.
        // El truco aqui es que cada hilo lleva sus propias sumas locales
        // para no tener que pelearse con los otros hilos por escribir
        // en el mismo lugar. Al final se juntan todas en una seccion critica.
        std::vector<double> suma_x(K, 0.0);
        std::vector<double> suma_y(K, 0.0);
        std::vector<int> conteo(K, 0);

        #pragma omp parallel
        {
            // sumas privadas de cada hilo, nadie mas las toca
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

            // aqui cada hilo aporta su parte al total global.
            // solo un hilo a la vez puede entrar aqui (seccion critica).
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    suma_x[k] += local_x[k];
                    suma_y[k] += local_y[k];
                    conteo[k] += local_conteo[k];
                }
            }
        }

        // ya con las sumas completas, calculamos la nueva posicion de cada centroide
        for (int k = 0; k < K; k++) {
            if (conteo[k] > 0) {
                centroides[k].x = suma_x[k] / conteo[k];
                centroides[k].y = suma_y[k] / conteo[k];
            }
        }
    }

    return iter;
}

// Punto de entrada del programa. Recibe 4 argumentos:
// 1. archivo CSV de entrada
// 2. K (numero de clusters)
// 3. archivo CSV de salida
// 4. numero de hilos a usar
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