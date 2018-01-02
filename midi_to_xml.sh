#!/bin/bash
for filename in /Users/faraaz/workspace/apollo/data/midi/**/*.mid; do
	/Applications/MuseScore\ 2.app/Contents/MacOS/mscore -o "${filename%.mid}.xml" "$filename"
done