from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

t = np.arange(0, 8.5, 0.5)
# input
plt.subplot(4, 1, 1)
y1 = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]
plt.plot(t, y1, drawstyle='steps-post')
plt.xlim(0, 8)
plt.ylim(-0.5, 1.5)
plt.title('Input Signal')

# I Signal
plt.subplot(4, 1, 2)
a = 1 / np.sqrt(2)
tI = np.arange(0, 9, 1)
yI = [-a, a, -a, a, -a, a, -a, a, a]
plt.plot(tI, yI, drawstyle='steps-post')
plt.xlim(0, 8)
plt.ylim(-2, 2)
plt.title('I signal')

# Q signal
plt.subplot(4, 1, 3)
yQ = [a, -a, -a, a, a, -a, -a, a, a]
plt.plot(tI, yQ, drawstyle='steps-post')
plt.xlim(0, 8)
plt.ylim(-1, 1)
plt.title('Q Signal')

# QPSK signal
plt.subplot(4, 1, 4)
t = np.arange(0, 9., 0.01)


def outputwave(I, Q, t):
    rectwav = []
    for i in range(len(I)):
        t_tmp = t[((i) * 100):((i + 1) * 100)]
        yI_tmp = yI[i] * np.ones(100)
        yQ_tmp = yQ[i] * np.ones(100)
        wav_tmp = yI_tmp * np.cos(2 * np.pi * 5 * t_tmp) - yQ_tmp * np.sin(2 * np.pi * 5 * t_tmp)
        rectwav.append(wav_tmp)
    return rectwav


rectwav = outputwave(yI, yQ, t)
plt.plot(t, np.array(rectwav).flatten(), 'r')
plt.xlim(0, 8)
plt.ylim(-2, 2)
plt.title('QPSK Signal')

plt.tight_layout()
plt.show()
