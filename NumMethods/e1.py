import numpy as np
import matplotlib.pyplot as plt

def euler(f, x0, y0, h, n):
    """
    Método de Euler (explícito) para y' = f(x, y).
    f  : función f(x,y)
    x0 : x inicial
    y0 : y(x0)
    h  : tamaño de paso
    n  : número de pasos
    return: (x_vals, y_vals)
    """
    x_vals = np.zeros(n+1)
    y_vals = np.zeros(n+1, dtype=float)

    x_vals[0], y_vals[0] = x0, y0
    for i in range(n):
        y_vals[i+1] = y_vals[i] + h * f(x_vals[i], y_vals[i])
        x_vals[i+1] = x_vals[i] + h
    return x_vals, y_vals

# Ejemplo: y' = x + y, y(0) = 1
f = lambda x, y: x + y
x0, y0 = 0.0, 1.0
h      = 0.1
n      = 50

x, y_num = euler(f, x0, y0, h, n)

# Solución exacta para comparar: y(x) = -x - 1 + 2*e^x
y_exact = -x - 1 + 2*np.exp(x)

# Gráfica
plt.figure(figsize=(8,5))
plt.plot(x, y_num, 'o-', label='Euler (numérico)')
plt.plot(x, y_exact, '--', label='Exacta')
plt.xlabel('x'); plt.ylabel('y(x)')
plt.title('Método de Euler (explícito)')
plt.grid(True); plt.legend()
plt.show()

# (Opcional) error punto a punto
err = np.abs(y_exact - y_num)
print(f"Error máximo = {err.max():.3e}")
