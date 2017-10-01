'''
Extract pitch (F0)

Source:
- https://github.com/jaekookang/useful_bits/blob/master/Speech/Extract_Pitch_using_Praat/Extract_Pitch.ipynb

Usage:
get_pitch('da_ta.wav', 0.5)
> 118.177

'''

import os
import numpy as np
from subprocess import Popen, PIPE
import pdb

tmp_script = 'tmp.praat'
def gen_script():
    # This generates temporary praat script file
    global tmp_script
    with open(tmp_script, 'w') as f:
        f.write('''
form extract_pitch
text FILENAME
positive TIMEAT 0.0
positive TIMESTEP 0.0
real FLOOR 75.0
real CEILING 600.0
endform
Read from file... 'FILENAME$'
To Pitch... 'TIMESTEP' 'FLOOR' 'CEILING'
Get value at time... 'TIMEAT' Hertz Linear
exit
''')
    return tmp_script
        
def run_praat_cmd(*args):
    o = Popen(['praat'] + [str(i) for i in args],
             shell=False, stdout=PIPE, stderr=PIPE)
    stdout, stderr = o.communicate()
    if os.path.exists(tmp_script): 
        os.remove(tmp_script)
    if o.returncode:
        raise Exception(stderr.decode('utf-8'))
    else:
        return stdout
        
def get_pitch(FNAME, TIMEAT, TIMESTEP=0.0, FLOOR=75.0, CEILING=600.0):
    def _float(s):
        # Retrieved from https://github.com/mwv/praat_formants_python
        try:
            return float(s)
        except ValueError:
            return np.nan
    out = run_praat_cmd(gen_script(), FNAME, TIMEAT, TIMESTEP, FLOOR, CEILING)
    outstr = str(out, 'utf-8').split()
    if len(outstr) < 2:
        print('--undefined--')
        val = 0.0 # pad nan as 0
    else:
        val = float('{:.3f}'.format(float(outstr[0])))
    return val

# time = 0.5 # sec
# get_pitch('da_ta.wav', time) # output: F0