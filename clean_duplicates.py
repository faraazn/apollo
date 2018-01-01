import os
import glob

for filename in glob.glob('data/midi/**/*', recursive=True):
	if ' ' in filename:
		print(filename)
		os.remove(filename)