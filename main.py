import random
from random import randint
from statistics import mean

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

length = 10000  # длина
c = 1500  # Скорость звука
h = 100  # Глубина 100м
l = 1  # Номер моды
fc = 100  # Несущая частота

w = []
w_freq = []
k = []
q = []
fi = []  # фазовый набег
new_w = []
zeroes = int(length / c * (1 / sampling_t)) * 2
# zeroes = 0

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
        fi.append(complex(q[i3] * length))
    else:
        fi.append(complex(-1 * q[i3] * length))



def regular_signal():
    # time_for_regular_singal = np.arange(0, (30000) / sampling_freq, 1000)
    coef = 2
    k_regular = []
    q_regular = []
    fi_regular = []
    w_regular = []
    w_new_regular = []
    length_regular = 10000
    l_moda = 1
    h_regular = 300 # глубина
    fc_regular = 300  # Несущая частота
    zeroes_regular = int(length_regular / c * (1 / sampling_t)) * 2
    for i_regular in range(1000):  # 1000 отсчётов на 0.5 сек
        if i_regular == 0:
            w_regular.append(0)
        else:
            w_regular.append(w_regular[i_regular - 1] + sampling_t)

    for i_new_regular in range(1000 + zeroes_regular):
        if i_new_regular == 0:
            w_new_regular.append(0)
        else:
            w_new_regular.append(w_new_regular[i_new_regular - 1] + sampling_t)

    w_reqular_freq = []
    for i_regular_freq in range(1000 + zeroes_regular):
        if i_regular_freq == 0:
            w_reqular_freq.append(0)
        else:
            w_reqular_freq.append(w_reqular_freq[i_regular_freq - 1] + sampling_freq / (1000 + zeroes_regular - 1))


    regular_signal = []
    for i_regular_2 in range(1000):  # 1000 отсчётов на 0.5 сек
        e = math.e ** (-coef * w_regular[i_regular_2])
        sig_part = math.sin(2 * pi * fc_regular * w_regular[i_regular_2])
        regular_signal.append(e * sig_part)

    fig_3 = plt.figure(figsize=(10, 10))
    ax21 = fig_3.add_subplot(2, 1, 1)
    ax21.set_title('Исходный сигнал', fontsize=20)
    plt.axis([0, 0.5, -1.2, 1.2])
    plt.plot(w_regular, regular_signal, 'c')
    plt.xlabel('Время, сек.', fontsize=10)
    plt.ylabel('Давление, произв. единицы', fontsize=10)

    for i_regular_3 in range(len(w_reqular_freq)):
        if i_regular_3 <= len(w_reqular_freq) / 2:
            k_regular.append(2 * pi * w_reqular_freq[i_regular_3] / c)
        else:
            k_regular.append(2 * pi * w_reqular_freq[len(w_reqular_freq) - i_regular_3] / c)

    for i_regular_4 in range(len(k_regular)):
        q_regular.append((k_regular[i_regular_4] ** 2 - pi ** 2 / h_regular ** 2 * (l_moda - 0.5) ** 2) ** 0.5)


    for i_regular_5 in range(len(q_regular)):
        if i_regular_5 <= len(q_regular) / 2:
            fi_regular.append(q_regular[i_regular_5] * length_regular)
        else:
            fi_regular.append(-1 * q_regular[i_regular_5] * length_regular)

    for i_regular_zeroes in range(zeroes_regular):  # Добивание нулями
        regular_signal.append(0)

    regular_signal_fft = scipy.fft.fft(regular_signal)

    regular_zeroes = 0
    for i_regular_zer in range(int(len(regular_signal_fft) / 2)):  # занулене
        if isinstance(q_regular[i_regular_zer], complex):  # если значение q для значения спектра излучённого сигнала явл комплексным, то значение спектра зануляется
            regular_signal_fft[i_regular_zer] = 0
        if 2 * pi * w_new_regular[i_regular_zer] / q_regular[i_regular_zer].real + 0.0000000001 > 1700:
            regular_signal_fft[i_regular_zer] = 0
        if regular_signal_fft[i_regular_zer] == 0:
            regular_zeroes += 1

    for i_regular_zer_2 in range(len(regular_signal_fft)):
        if i_regular_zer_2 > len(regular_signal_fft) - regular_zeroes:
            regular_signal_fft[i_regular_zer_2] = 0


    regular_sign = []
    for i_regular_6 in range(len(regular_signal_fft)):
        if isinstance(fi_regular[i_regular_6], complex):
            fi_var = 0
        else:
            fi_var = fi_regular[i_regular_6]
        regular_sign.append(regular_signal_fft[i_regular_6] * pow(math.e, (-1j * fi_var))) # Спектр принятого сигнала

    regular_sign_ifft = scipy.fft.ifft(regular_sign)  # Принятый сигнал после обратного преобразования Фурье



    ax22 = fig_3.add_subplot(2, 1, 2)
    ax22.set_title('Принятый сигнал', fontsize=20)
    plt.axis([length_regular / c - 0.01, length_regular / c + 0.6, -1.2, 1.2])
    plt.plot(w_new_regular, regular_sign_ifft, 'g')
    plt.xlabel('Время, сек.', fontsize=10)
    plt.ylabel('Давление, произв. единицы', fontsize=10)

    plt.text(length_regular / c, -1.9, 'Частота = ' + str(fc_regular) + "Гц "
                                        'Длина передачи = ' + str(length_regular) + "м \n"
                                        + 'Глубина = ' + str(h_regular) + " м "
                                        + "мода " + str(l_moda), fontsize=20)
    plt.show()





# Random generate signal sequence
a = np.random.randint(0, 2, size * 20)  # массив значений 0 и 1
m = np.zeros(len(w), dtype=np.float32)

a = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]

for i in range(len(w)):
    m[i] = a[math.floor(w[i] * 20)]

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    if isinstance(q[i_fft],
                  complex):  # если значение q для значения спектра излучённого сигнала явл комплексным, то значение спектра зануляется
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

noise = []
noise_diapazon = 1.75
for i_noise in range(n):  # заполнение массива шума
    noise.append(random.uniform(-noise_diapazon, noise_diapazon))

#noise = []
#
#f = open('random_noise.txt', 'a')
#for i_rand in range(len(noise)):
#    f.write(str(noise[i_rand]) + ", ")
#f.close()


noise_avg = 0
for i_noise_avg in range(len(noise)):
    noise_avg += noise[i_noise_avg] ** 2
noise_avg = math.sqrt(noise_avg)  # среднеквадратичное значение шума

sign_ifft_avg = 0
for i_sign_ifft_avg in range(len(sign_ifft)):
    if i_sign_ifft_avg > length / c * sampling_freq and i_sign_ifft_avg < length / c * sampling_freq + size * sampling_freq:
        sign_ifft_avg += sign_ifft[i_sign_ifft_avg] ** 2
sign_ifft_avg = math.sqrt(sign_ifft_avg.real)  # среднеквадратичное значение принятого сигнала

signal_noise = round(10 * math.log10(sign_ifft_avg / noise_avg + 0.00000001), 3)
print(signal_noise)

counter = 0
#for i_noise_add in range(len(sign_ifft)):  # наложение шума на принятый сигнал
#    if i_noise_add > length / c * sampling_freq and i_noise_add < length / c * sampling_freq + size *sampling_freq:
#        sign_ifft[i_noise_add] += noise[counter]
#        counter += 1

sign_ifft_demodulation = []
for i_demodulation in range(len(sign_ifft)):
    sign_ifft_demodulation.append(sign_ifft[i_demodulation] * np.cos(2 * pi * fc * new_w[i_demodulation]))

temp = 0
method_of_avg = []

for i_demodulation_2 in range(len(sign_ifft_demodulation)):  # метод скользящей средней
    if i_demodulation_2 + 50 < len(sign_ifft_demodulation):
        for i_ten in range(50):
            temp += sign_ifft_demodulation[i_demodulation_2 + i_ten]
    else:
        for i_ten in range(50):
            temp += sign_ifft_demodulation[i_demodulation_2 - i_ten]
    method_of_avg.append(temp / 50)
    temp = 0

demodulated_signal = []
for i_demodulation_3 in range(len(method_of_avg)):
    if method_of_avg[i_demodulation_3] >= 0.01:
        demodulated_signal.append(1)
    else:
        demodulated_signal.append(0)

counter = 0
amount = 0
final_array = []
temp_final_array = []
zero_or_one = demodulated_signal[int(length / c * sampling_freq) + 1]

hundred = 0
for i_demodulation_4 in range(size * 20):
    final_array.append(demodulated_signal[int(length / c * sampling_freq) + 50 + hundred])
    hundred += 100

yes = 0
no = 0
for i_checking in range(len(final_array)):
    if final_array[i_checking] == a[i_checking]:
        yes += 1
    else:
        no += 1

for teast_i in range(len(final_array)):
    if final_array[teast_i] == a[teast_i]:
        answer = "YES"
    else:
        answer = "NO"
#    print(str(a[teast_i]) + "  " + str(final_array[teast_i]) + " : " + answer)

# TODO рассчитать начало сигнала и не опираться на то, что для скользящей средней взято малое кол-во отсчётов


print("Количество рандомно сгенерированных значений: " + str(len(a)))
print("Совпавшие значения: " + str(yes))
print("Ошибки: " + str(no))
ber = round(no / (size * 20) * 100, 2)
#ber = 10 * math.log10(ber)
print("Вероятность битовой ошибки: " + str(ber) + "%")
if length == 10000:
    f = open('BER_10km.txt', 'a')
elif length == 30000:
    f = open('BER_30km.txt', 'a')
try:
    if length == 10000 or length == 30000:
        f.write(str(ber) + "\n")
        f.write(str(signal_noise) + "\n")
finally:
    if length == 10000 or length == 30000:
        f.close()

start = 0
finish = 0.3

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(7, 1, 1)
ax1.set_title('Случайный бинарный сигнал', fontsize=20)
plt.axis([start, finish, -0.5, 1.5])
plt.plot(w, m, 'b')
fig.tight_layout(h_pad=3)

ax2 = fig.add_subplot(7, 1, 2)
ax2.set_title('BPSK модулированный сигнал', fontsize=20)
plt.axis([start, finish, -1.5, 1.5])
plt.plot(new_w, bpsk, 'r')

ax3 = fig.add_subplot(7, 1, 3)
ax3.set_title('Принятый сигнал', fontsize=20)
plt.axis([start + length / c, length / c + finish, -2, 2])
plt.plot(new_w, sign_ifft, 'g')

ax4 = fig.add_subplot(7, 1, 4)
ax4.set_title('Принятый сигнал после умножения на несущий',fontsize=20)
plt.axis([start + length / c, length / c + finish, -2, 2])
plt.plot(new_w, sign_ifft_demodulation, 'm')

ax5 = fig.add_subplot(7, 1, 5)
ax5.set_title('Метод скользящей средней', fontsize=20)
plt.axis([start + length / c, length / c + finish, -2, 2])
plt.plot(new_w, method_of_avg, 'y')

ax6 = fig.add_subplot(7, 1, 6)
ax6.set_title('Демодулированный сигнал', fontsize=20)
plt.axis([start + length / c, length / c + finish, -0.5, 1.5])
plt.plot(new_w, demodulated_signal, 'c')

plt.text(start + length / c, -2.8, 'Частота = ' + str(fc) + "Гц " 
                                'Длина передачи = ' + str(length) + "м "
                                + 'Глубина = ' + str(h) + "м \n"
                                + "мода " + str(l),  fontsize=20)

#//////////////////////////////////////////////////////////////////////////////////
if length == 10000 or length == 30000:
    fig_2 = plt.figure(figsize=(10, 10))

    f = open('BER_10km.txt', 'r')
    try:
        ber_10_x_y = f.readlines()
    finally:
        f.close()
    ber_10_x = []
    ber_10_y = []
    for i_ber_10 in range(len(ber_10_x_y)):
        if i_ber_10 % 2 == 0:
            ber_10_x.append(float(ber_10_x_y[i_ber_10]))
        else:
            ber_10_y.append(float(ber_10_x_y[i_ber_10]))
    f = open('BER_30km.txt', 'r')
    try:
        ber_30_x_y = f.readlines()
    finally:
        f.close()
    ber_30_x = []
    ber_30_y = []
    for i_ber_30 in range(len(ber_30_x_y)):
        if i_ber_30 % 2 == 0:
            ber_30_x.append(float(ber_30_x_y[i_ber_30]))
        else:
            ber_30_y.append(float(ber_30_x_y[i_ber_30]))


    ax11 = fig_2.add_subplot(2, 1, 1)
    ax11.set_title("Отношение сигнал/шум к BER на 10км", fontsize=20)
    plt.axis([-10, 0, 0, 25])
    plt.plot(ber_10_y, ber_10_x, 'c')
    plt.xlabel('отношение сигнал/шум, Дб', fontsize=10)
    plt.ylabel('Вероятность битовой ошибки в %', fontsize=10)


    ax12 = fig_2.add_subplot(2, 1, 2)
    ax12.set_title('Отношение сигнал/шум к BER на 30км', fontsize=20)
    plt.axis([-10, 0, 0, 25])
    plt.plot(ber_30_y, ber_30_x, 'c')
    plt.xlabel('отношение сигнал/шум, Дб', fontsize=10)
    plt.ylabel('Вероятность битовой ошибки в %', fontsize=10)

plt.show()
regular_signal()

