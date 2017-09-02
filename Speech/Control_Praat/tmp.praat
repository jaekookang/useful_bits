
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
