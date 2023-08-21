import matplotlib.pyplot as plt
import numpy as np

doa = np.pi / 4
_range = 20
space = 0.5
m_sensors = 20
x = _range * np.sin(doa)
y = _range * np.cos(doa)
sensors_x = np.linspace(0, m_sensors - 1, m_sensors) * space - space * int(m_sensors / 2)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

z = np.linspace(0, doa)
_r = 2
plt.plot(_r * np.sin(z), _r * np.cos(z), label="DOA")

plt.plot([0, x], [0, y], "--", label="Range")
plt.plot(sensors_x, np.zeros(m_sensors), "o", label="Array")
plt.plot([x], [y], "v", label="Target")
plt.grid()
plt.legend()

plt.show()
