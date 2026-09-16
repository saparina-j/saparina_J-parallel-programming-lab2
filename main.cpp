#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <iomanip>

using namespace std;

vector<vector<double>> multiplySequential(const vector<vector<double>>& A,
                                          const vector<vector<double>>& B,
                                          int n) {
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<double>> multiplyParallel(const vector<vector<double>>& A,
                                         const vector<vector<double>>& B,
                                         int n) {
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> M(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = (double)(rand() % 100) / 10.0;
        }
    }
    return M;
}

double maxError(const vector<vector<double>>& A,
                const vector<vector<double>>& B) {
    double maxErr = 0.0;
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            maxErr = max(maxErr, fabs(A[i][j] - B[i][j]));
        }
    }
    return maxErr;
}

void runExperiment(int n, int numThreads, bool verify) {
    srand(42);
    auto A = generateMatrix(n);
    auto B = generateMatrix(n);

    omp_set_num_threads(numThreads);

    double startTime = omp_get_wtime();
    auto C_parallel = multiplyParallel(A, B, n);
    double endTime = omp_get_wtime();

    double elapsed = endTime - startTime;

    double operations = 2.0 * n * n * n;
    double gflops = (operations / elapsed) / 1e9;

    cout << fixed << setprecision(4);
    cout << "Size: " << setw(5) << n
         << " | Threads: " << setw(2) << numThreads
         << " | Time: " << setw(10) << elapsed << " sec"
         << " | GFLOPS: " << setw(8) << gflops;

    if (verify && n <= 400) {
        auto C_sequential = multiplySequential(A, B, n);
        double err = maxError(C_parallel, C_sequential);
        cout << " | Error: " << scientific << setprecision(2) << err;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
        cout << "OpenMP version: " << _OPENMP << endl;
    #else
        cout << "OpenMP NOT SUPPORTED!" << endl;
        return 1;
    #endif

    cout << "Available processors: " << omp_get_num_procs() << endl;
    cout << "Max threads: " << omp_get_max_threads() << endl;
    cout << string(80, '-') << endl;

    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<int> threadCounts = {1, 2, 4, 6, 8, 12};

    cout << "\n=== VERIFICATION ===" << endl;
    runExperiment(200, 4, true);
    cout << endl;

    cout << "=== EXPERIMENTS ===" << endl;
    cout << string(80, '-') << endl;

    for (int numThreads : threadCounts) {
        cout << "\n--- Threads: " << numThreads << " ---" << endl;
        for (int n : sizes) {
            runExperiment(n, numThreads, false);
        }
    }

    return 0;
}