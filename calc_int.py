import numpy as np

def f(u, R, d):
    r = R + d * u
    return (R + d * u) * np.sin(np.pi/2 * (1 - np.tanh(u))) / np.cosh(u)**2

def trapezoidal_integration(a, b, n, R, d):
    h = (b - a) / n
    s = 0.5 * (f(a, R, d) + f(b, R, d))
    for i in range(1, n):
        s += f(a + i * h, R, d)
    return s * h * d**2

# Parameters
R = 1.0
d = 0.5
n = 1000  # number of intervals

# Limits of integration
a = -R / d
b = R / d

# Compute the integral
integral = trapezoidal_integration(a, b, n, R, d)
print("The computed integral using the trapezoidal rule is:", integral)