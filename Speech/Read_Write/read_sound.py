# coding: utf-8

# Read sound file
# python3.5
# 2017-04-01
# ref: http://samcarcagno.altervista.org/blog/basic-sound-processing-python/

from pylab import*
from scipy.io import wavfile
import matplotlib.pyplot as plt

srate, sig = wavfile.read('da_ta.wav')
duration = len(sig)/srate

print('srate(Hz):', srate)
print('duration(sec):',duration)

plt.plot(sig)
plt.show()

