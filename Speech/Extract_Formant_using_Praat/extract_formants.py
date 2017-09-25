import os
import numpy as np
from subprocess import Popen, PIPE
from bisect import bisect_left

tmp_script = 'tmp.praat'
def gen_script():
    # This generates temporary praat script file
    global tmp_script
    with open(tmp_script, 'w') as f:
        f.write('''
form extract_formant
text FILENAME
positive MAXFORMANT 5500
real WINLEN 0.025
positive PREEMPH 50
endform
Read from file... 'FILENAME$'
To Formant (burg)... 0.01 5 'MAXFORMANT' 'WINLEN' 'PREEMPH'
List... no yes 6 no 3 no 3 no
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
        
def get_formant(FNAME, time, MAXFORMANT=5500, WINLEN=0.025, PREEMPH=50):
    fmt_out = {}
    def _float(s):
        # Retrieved from https://github.com/mwv/praat_formants_python
        try:
            return float(s)
        except ValueError:
            return np.nan
    key = (FNAME, MAXFORMANT, WINLEN, PREEMPH)
    run_out = run_praat_cmd(gen_script(), FNAME, MAXFORMANT, WINLEN, PREEMPH)
    fmt_out[key] = np.array(list(map(lambda x: list(map(_float, x.rstrip().split('\t')[:4])), 
                                     run_out.decode('utf-8').split('\n')[1:-1])))
    out = fmt_out[key]
    val = out[bisect_left(out[:,0], time), 1:]
    if np.any(np.isnan(val)):
        val = 0.0 # pad nan as 0
    return val

# time = 0.5 # sec
# get_formant('da_ta.wav', time) # output: F1, F2, F3