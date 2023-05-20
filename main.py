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

length = 30000  # длина
c = 1500  # Скорость звука
h = 100  # Глубина 100м
l = 1  # Номер моды

w = []
w_freq = []
k = []
q = []
fi = []  # фазовый набег
new_w = []
zeroes = int(length / c * (1 / sampling_t)) * 2
#zeroes = 0

for i0 in range(n):
    if i0 == 0:
        w.append(0)
    else:
        w.append(w[i0 - 1] + sampling_t)

for i_new_w in range(n + zeroes):
    if i_new_w == 0:
        new_w.append(0)
    else:
        new_w.append(new_w[i_new_w - 1] + sampling_t)

for iw_freq in range(n + zeroes):
    if iw_freq == 0:
        w_freq.append(0)
    else:
        w_freq.append(w_freq[iw_freq - 1] + sampling_freq / (n + zeroes - 1))

for i1 in range(len(w_freq)):
    if i1 <= len(w_freq) / 2:
        k.append(2 * pi * w_freq[i1] / c)
    else:
        k.append(2 * pi * w_freq[len(w_freq) - i1] / c)

for i2 in range(len(k)):
    q.append((k[i2] ** 2 - pi ** 2 / h ** 2 * (l - 0.5) ** 2) ** 0.5)

for i3 in range(len(q)):
    if i3 <= len(q) / 2:
        fi.append(q[i3] * length)
    else:
        fi.append(-1 * q[i3] * length)

# Random generate signal sequence
a = np.random.randint(0, 2, size * 20)  # массив значений 0 и 1
m = np.zeros(len(w), dtype=np.float32)

for i in range(len(w)):
    m[i] = a[math.floor(w[i] * 20)]



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
fc = 100  # Несущая частота
ts = np.arange(0, (30000) / sampling_freq, 1 / sampling_freq)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))

bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)
bpsk = list(bpsk)


for i_zeroes in range(zeroes):  # Добивание нулями
    bpsk.append(0)

bpsk_fft = scipy.fft.fft(bpsk)  # Сигнал bpsk после прямого преобразования Фурье
# print(len(bpsk_fft))


bpsk_zeroes = 0

for i_fft in range(int(len(bpsk_fft) / 2)):  # занулене
    if isinstance(q[i_fft], complex): # если значение q для значения спектра излучённого сигнала явл комплексным, то значение спектра зануляется
        bpsk_fft[i_fft] = 0
    if 2 * pi * w_freq[i_fft] / q[i_fft].real + 0.0000000001 > 1700:
        bpsk_fft[i_fft] = 0
    if bpsk_fft[i_fft] == 0:
        bpsk_zeroes += 1

for i_fft_2 in range(len(bpsk_fft)):
    if i_fft_2 > len(bpsk_fft) - bpsk_zeroes:
        bpsk_fft[i_fft_2] = 0

sign = []

for i4 in range(len(bpsk_fft)):
    sign.append(bpsk_fft[i4] * (math.e ** (-1j * fi[i4])))  # Спектр принятого сигнала
    if i4 == len(bpsk_fft) / 2:
        sign[i4] = 0


sign_ifft = scipy.fft.ifft(sign)  # Принятый сигнал после обратного преобразования Фурье
sign_ifft_demodulation = []

for i_demodulation in range(len(sign_ifft)):
   sign_ifft_demodulation.append(sign_ifft[i_demodulation] * -np.cos(2 * pi * fc * new_w[i_demodulation]))

temp = 0

for i_demodulation_2 in range(len(sign_ifft_demodulation)):
    if i_demodulation_2 + 50 < len(sign_ifft_demodulation):
        for i_ten in range(50):
            temp += sign_ifft_demodulation[i_demodulation_2 + i_ten]
    else:
        for i_ten in range(50):
            temp += sign_ifft_demodulation[i_demodulation_2 - i_ten]
    sign_ifft_demodulation[i_demodulation_2] = temp / 50
    temp = 0




# sign_ifft = np.real(sign_ifft)

#for i_printer in range(len(sign_ifft)):
#   if i_printer < 3:
#       print(str(i_printer + 1) + ": " + str(sign_ifft[i_printer]))
#   if len(sign_ifft) / 2 - 2 < i_printer < len(sign_ifft) / 2 + 2:
#       print(str(i_printer + 1) + ": " + str(sign_ifft[i_printer]))
#   if i_printer > len(sign_ifft) - 4:
#       print(str(i_printer + 1) + ": " + str(sign_ifft[i_printer]))

#for i_temp in range(len(sign_ifft)):
#    if  40000 > i_temp > 20000:
#        print(str(i_temp) + ":" + str(sign_ifft[i_temp]))

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(4, 1, 1)
ax1.set_title('generate Random Binary signal', fontsize=20)
plt.axis([0, 0.3, -0.5, 1.5])
plt.plot(w, m, 'b')
fig.tight_layout(h_pad=2.5)


ax2 = fig.add_subplot(4, 1, 2)
ax2.set_title('BPSK Modulation', fontsize=20)  # , fontproperties=zhfont1
plt.axis([0, 0.3, -1.5, 1.5])
plt.plot(new_w, bpsk, 'r')

ax3 = fig.add_subplot(4, 1, 3)
ax3.set_title('Принятый сигнал' + " (" + str(length) + "м)", fontsize=20)  # , fontproperties=zhfont1
plt.axis([length / c, length / c + 0.3, -2, 2])
plt.plot(new_w, sign_ifft, 'g')

ax4 = fig.add_subplot(4, 1, 4)
ax4.set_title('Демодуляция' + " (" + str(length) + "м)", fontsize=20)
plt.axis([length / c, length / c + 0.3, -2, 2])
plt.plot(new_w, sign_ifft_demodulation, 'y')
plt.show()
