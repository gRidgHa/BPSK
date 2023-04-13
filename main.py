
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import math
import scipy


# Number

size = 10
sampling_t = 0.01  # шаг в 0.01 секунду
w = np.arange(0, size, sampling_t)
length = 1500  # длина
c = 1500  # Скорость звука
h = 100  # Глубина 100м
l = 1  # Номер моды

k = []
q = []
for i1 in range(len(w)):
    k.append(2 * pi * w[i1] / c)
#for i2 in range(len(k)):
#    q.append((np.sign(k[i2]) ^ 2 - pi ** 2 / h ** 2 * (l - 0.5) ** 2) ** 0.5)

print(k)

# Random generate signal sequence
a = np.random.randint(0, 2, size)
m = np.zeros(len(w), dtype=np.float32)
for i in range(len(w)):
    m[i] = a[math.floor(w[i])]
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)

ax1.set_title('generate Random Binary signal', fontsize=20)
plt.axis([0, size, -0.5, 1.5])
plt.plot(w, m, 'b')
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
fc = 4000
fs = 20 * fc  # Sampling frequency
ts = np.arange(0, (100 * size) / fs, 1 / fs)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))

bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)
bpsk_fft = scipy.fft.fft(bpsk)  # Сигнал bpsk после прямого преобразования Фурье
# print(bpsk_fft)

# plt.axis([0, size, -1.5, 1.5])
# plt.plot(t, bpsk_fft)
# plt.show()

# BPSK modulated signal waveform
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('BPSK Modulation', fontsize=20)  # , fontproperties=zhfont1
plt.axis([0, size, -1.5, 1.5])
plt.plot(w, bpsk, 'r')
plt.show()
