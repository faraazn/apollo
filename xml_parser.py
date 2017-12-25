import os
import music21
import numpy as np

corpus = ['/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m15791_5.xml',
			'/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m5937_8.xml',
			'/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m15715_5.xml',
			'/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m15714_5.xml',
			'/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m17146_3.xml']

MEASURES_PER_CUT = 32
MAX_NOTE = music21.note.Note('C8')
MIN_NOTE = music21.note.Note('A0')
MAX_PITCH = MAX_NOTE.pitches[0].midi
MIN_PITCH = MIN_NOTE.pitches[0].midi
assert MAX_PITCH == 108
assert MIN_PITCH == 21
NOTE_RANGE = int(MAX_PITCH - MIN_PITCH + 1)
TIME_SIGNATURE = music21.meter.TimeSignature('2/4')
GRANULARITY = 16
stats = {'discarded_less_measures': 0, 
			'discarded_time_signature': 0, 
			'discarded_note_range': 0,
			'discarded_more_parts': 0,
			'discarded_high_granularity': 0,
			'total_cut_scores': 0,
			'total_scores': 0,
			'total_measures': 0}

def encode_score(score):
	assert len(score.parts) == 2
	score = score.flattenParts()
	score_X = []
	start_ind = 1
	cut_score = score.measures(start_ind, start_ind+MEASURES_PER_CUT-1)
	measures = cut_score.getElementsByClass('Measure')
	while len(measures) == MEASURES_PER_CUT:
		for ts in cut_score.getTimeSignatures():
			assert ts.ratioString == TIME_SIGNATURE.ratioString
			print(ts.ratioString)
		
		x = np.zeros((MEASURES_PER_CUT*GRANULARITY*TIME_SIGNATURE.beatCount, NOTE_RANGE))
		for measure in measures:
			notes = measure.getElementsByClass('Note')
			for note in notes:
				if note.quarterLength < 4.0 / GRANULARITY:
					continue
				pitch = note.pitches[0].midi
				if pitch < MIN_PITCH or pitch > MAX_PITCH:
					continue
				ind = measure.measureNumber % MEASURES_PER_CUT
				ind *= int(GRANULARITY*TIME_SIGNATURE.beatCount)
				ind += int(note.offset * GRANULARITY)
				for i in range(int(note.quarterLength * GRANULARITY / 4)):
					x[ind+i][pitch-MIN_PITCH] = 1
		print(x[0])
		print(x[1])
		print(x[2])
		score_X.append(x)
		start_ind += MEASURES_PER_CUT
		cut_score = score.measures(start_ind, start_ind+MEASURES_PER_CUT-1)
		measures = cut_score.getElementsByClass('Measure')
	return score_X

X = []
Y = []
for song_path in corpus:
	try: 
		score = music21.converter.parse(song_path)
		x = encode_score(score)
		X.append(x)
		Y.append('mozart')
		print("Encoded ", song_path)
	except AssertionError:
		pass
	except ZeroDivisionError:
		pass