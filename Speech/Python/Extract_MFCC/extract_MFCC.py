# Extract MFCC from a sound
# 2017-04-02 jkang
# Python3.5
#
# **Prerequisite**
# - Install python_speech_features
#   >> https://github.com/jameslyons/python_speech_features

from pylab import*
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

# Read sound
srate, sig = wavfile.read('da_ta.wav')
plt.plot(np.arange(len(sig))/srate, sig)
plt.title('da_ta.wav')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.show()

# Extract MFCC
winlen = 0.025
winstep = 0.01
numcep = 13
mfcc_raw = mfcc(sig, srate, winlen, winstep, numcep, appendEnergy = True) # 13-d MFCC
mfcc_deriv1 = delta(mfcc_raw, N = 2) # 1st deriv
mfccs = np.concatenate((mfcc_raw, mfcc_deriv1), axis=1).astype(np.float32)
plt.imshow(np.rot90(mfccs, axes=(0,1)), aspect='auto')
plt.title('MFCC values (26 dimension)')
plt.xlabel('Time (msec)')
plt.ylabel('Coefficients')
plt.show()

