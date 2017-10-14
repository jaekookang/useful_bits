'''
Plot real-time waveform

2017-10-01 jk

References:
- https://python-sounddevice.readthedocs.io/en/0.2.1/examples.html
'''

import math
import numpy as np
import shutil
import pdb
import matplotlib.pyplot as plt

class params():
    '''
    Prepare parameters
    '''
    def __init__(self):
        self.list_devices = True
        self.block_duration = 50
        self.device = None
        self.gain = 10
        self.freq_range = (100, 2000)

P = params()

try:
    import sounddevice as sd
    samplerate = sd.query_devices(P.device, 'input')['default_samplerate']
    
    # initialise the graph and settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    
    xval = []
    yval = []
    
    def callback(incoming, frames, time, status):
        if any(incoming):
            ax.clear()
            yval.append(incoming)
            ax.plot(yval, 'o-')
            fig.canvas.draw()

    with sd.InputStream(device=P.device, channels=1, callback=callback,
                        blocksize=int(samplerate * P.block_duration / 1000),
                        samplerate=samplerate):
        while True:
            response = input()
            if response in ('q'):
                break
    
except KeyboardInterrupt:
    print('Interrupted by user')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
        

# TODO: Fix problems
# TODO: Run in Terminal first
# TODO: Run in Jupyter later
