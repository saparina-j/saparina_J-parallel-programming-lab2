import matplotlib.pyplot as plt
import numpy as np

sizes = [200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 6, 8, 12]

gflops = {
    1:  [4.00, 4.27, 4.74, 3.81, 3.58, 3.59],
    2:  [8.00, 8.53, 9.57, 6.38, 6.30, 6.51],
    4:  [5.33, 12.80, 12.49, 12.71, 10.94, 10.89],
    6:  [8.00, 14.22, 18.62, 15.64, 14.92, 3.41],
    8:  [4.00, 4.41, 4.21, 3.71, 3.72, 5.28],
    12: [np.nan, 12.80, 8.90, 7.07, 4.11, 7.49],
}

times = {
    1:  [0.004, 0.030, 0.216, 0.907, 2.287, 4.462],
    2:  [0.002, 0.015, 0.107, 0.542, 1.300, 2.457],
    4:  [0.003, 0.010, 0.082, 0.272, 0.749, 1.469],
    6:  [0.002, 0.009, 0.055, 0.221, 0.549, 4.688],
    8:  [0.004, 0.029, 0.243, 0.932, 2.202, 3.033],
    12: [np.nan, 0.010, 0.115, 0.489, 1.995, 2.137],
}

# График 1: GFLOPS от размера матрицы
plt.figure(figsize=(10, 6))
for t in threads:
    plt.plot(sizes, gflops[t], 'o-', label=f'{t} threads')
plt.xlabel('Matrix size (n x n)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.title('OpenMP Matrix Multiplication: Performance', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gflops.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Ускорение от числа потоков
plt.figure(figsize=(10, 6))
for i, n in enumerate(sizes):
    speedup = [times[1][i] / times[t][i] if not np.isnan(times[t][i]) else np.nan
               for t in threads]
    plt.plot(threads, speedup, 'o-', label=f'n={n}')
plt.plot(threads, threads, 'k--', label='Ideal speedup')
plt.xlabel('Number of threads', fontsize=12)
plt.ylabel('Speedup (S = T1 / Tp)', fontsize=12)
plt.title('OpenMP Matrix Multiplication: Speedup', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speedup.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Время от размера матрицы
plt.figure(figsize=(10, 6))
for t in threads:
    plt.plot(sizes, times[t], 'o-', label=f'{t} threads')
plt.xlabel('Matrix size (n x n)', fontsize=12)
plt.ylabel('Time (sec)', fontsize=12)
plt.title('OpenMP Matrix Multiplication: Execution Time', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('time.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs saved: gflops.png, speedup.png, time.png")