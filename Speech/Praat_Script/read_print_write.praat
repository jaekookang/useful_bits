# This script does...
# 1. Read .wav files from a directory
# 2. Print file names in the info window
# 3. Write txt file including file names
#
# 2017-03-13 jk

my_dir$ = "/Users/jaegukang/Desktop/test"
result$ = my_dir$+"/result.txt"
Create Strings as file list... sound_list 'my_dir$'/*.wav
num_wavs = Get number of strings

clearinfo

for iwav from 1 to num_wavs
	fname$ = Get string... iwav
	printline 'iwav': 'fname$'
	fileappend 'result$' 'fname$' 'newline$'
endfor

select all
Remove