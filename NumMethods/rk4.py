import numpy as np
import matplotlib.pyplot as plt

def runge_kutta4(f, x0, y0, h, n):
    """
    Método de Runge-Kutta de 4to orden (RK4).
    
    f  : función f(x,y) que define dy/dx
    x0 : valor inicial de x
    y0 : valor inicial de y
    h  : tamaño de paso
    n  : número de pasos
    """
    x_vals = np.zeros(n+1)
    y_vals = np.zeros(n+1)
    
    x_vals[0], y_vals[0] = x0, y0
    
    for i in range(n):
        x, y = x_vals[i], y_vals[i]
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        
        y_vals[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x_vals[i+1] = x + h
    
    return x_vals, y_vals

# Ejemplo: dy/dx = x + y, con y(0)=1
f = lambda x, y: x + y
x0, y0 = 0, 1
h = 0.1
n = 50

x_vals, y_vals = runge_kutta4(f, x0, y0, h, n)

# Solución analítica: y(x) = -x - 1 + 2*e^x
y_exact = -x_vals - 1 + 2*np.exp(x_vals)

# Graficar resultados
plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, 'o-', label='RK4 (numérico)')
plt.plot(x_vals, y_exact, 'r--', label='Solución exacta')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Método de Runge-Kutta 4º orden')
plt.legend()
plt.grid(True)
plt.show()
