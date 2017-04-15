# Read TextGrid in Python
# 2017-04-02 jkang
# Python3.5
#
# **Prerequisite**
# - Install 'textgrid' package
#   from https://github.com/kylebgorman/textgrid

import textgrid
import numpy as np

T = textgrid.TextGrid()
T.read('stops.TextGrid')
w_tier = T.getFirst('phone').intervals # 'phone' tier

words_raw = []
for ival in range(len(w_tier)):
    words_raw.append(w_tier[ival].mark) # get labels
    print(w_tier[ival].mark)
    
# unique word list
words_list = list(set(words_raw))
print('unique words:', words_list)

