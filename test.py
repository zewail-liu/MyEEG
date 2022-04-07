from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.01)
y_sin = np.sin(x)
y_linear = 2 * x + 1
y_exp = np.exp(x)

plt.subplot(231)
plt.plot(x, y_sin)
plt.subplot(232)
plt.plot(x, y_linear)
plt.subplot(233)
plt.plot(x, y_exp)

s = np.array([y_sin, y_linear, y_exp])
ori = s
s = s.swapaxes(0, 1)
scaler = StandardScaler()
s = scaler.fit_transform(s)
s = s.swapaxes(0, 1)

plt.subplot(234)
plt.plot(x, s[0])
plt.subplot(235)
plt.plot(x, s[1])
plt.subplot(236)
plt.plot(x, s[2])
plt.show()
