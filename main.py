import numpy as np
from math import pi
import matplotlib.pyplot as plt
import math
import scipy

# Number

size = 15
sampling_t = 0.0005  # шаг в 0.0005 секунду
sampling_freq = 1 / sampling_t  # частота дискретизации
n = int(size / sampling_t)


w = []
w_freq = []
# w = np.arange(0, size, sampling_t)
for i0 in range(n):
    if i0 == 0:
        w.append(0)
    else:
        w.append(w[i0 - 1] + sampling_t)

for iw_freq in range(30000):
    if iw_freq == 0:
        w_freq.append(0)
    else:
        w_freq.append(w_freq[iw_freq - 1] + sampling_freq / n)


length = 15000  # длина
c = 1500  # Скорость звука
h = 100  # Глубина 100м
l = 1  # Номер моды

k = []
q = []
fi = []  # фазовый набег

for i1 in range(len(w_freq)):
    k.append(2 * pi * w_freq[i1] / c)

for i2 in range(len(k)):
    q.append((k[i2] ** 2 - pi ** 2 / h ** 2 * (l - 0.5) ** 2) ** 0.5)


for i3 in range(len(q)):
    fi.append(q[i3] * length)




# Random generate signal sequence
a = np.random.randint(0, 2, size * 20) # массив значений 0 и 1
m = np.zeros(len(w), dtype=np.float32)

for i in range(len(w)):
    m[i] = a[math.floor(w[i] * 20)]

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('generate Random Binary signal', fontsize=20)
plt.axis([0.9, 1.1, -0.5, 1.5])
plt.plot(w, m, 'b')

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
fc = 100  # Несущая частота
ts = np.arange(0, (30000) / sampling_freq, 1 / sampling_freq)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))

bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)
bpsk_fft = scipy.fft.fft(bpsk)  # Сигнал bpsk после прямого преобразования Фурье

for i_fft in range(len(bpsk_fft)):
    if isinstance(q[i_fft], complex):
        bpsk_fft[i_fft] = 0
    if 2 * pi * w_freq[i_fft] / q[i_fft].real + 0.0000000001 > 1700:
        bpsk_fft[i_fft] = 0

#for i_print in range(12000):
#    print(str(i_print) + ":" + str(bpsk_fft[i_print]) + " " + str(isinstance(q[i_print], complex)))
#print("7367 " + str(w_freq[7367]))
sign = []

for i4 in range(len(bpsk_fft)):
    sign.append(bpsk_fft[i4] * (math.e ** (-1j * fi[i4])))  # Спектр принятого сигнала
sign_ifft = scipy.fft.ifft(sign)  # Принятый сигнал после обратного преобразования Фурье



# plt.axis([0, size, -1.5, 1.5])
# plt.plot(t, bpsk_fft)
# plt.show()

# BPSK modulated signal waveform
#ax2 = fig.add_subplot(2, 1, 2)
#ax2.set_title('BPSK Modulation', fontsize=20)  # , fontproperties=zhfont1
#plt.axis([0.9, 1.1, -1.5, 1.5])
#plt.plot(w, bpsk, 'r')

ax3 = fig.add_subplot(2, 1, 2)
ax3.set_title('sign_ifft', fontsize=20)  # , fontproperties=zhfont1
plt.axis([0.9, 1.1, -1.5, 1.5])
plt.plot(w, sign_ifft, 'r')

plt.show()
