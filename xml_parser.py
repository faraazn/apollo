import os
import music21
import numpy as np

corpus = ['/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m5937_8.xml',
			'/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/m15791_5.xml',
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
TIME_SIGNATURE = music21.meter.TimeSignature('3/4') # TODO: what happens when denominator!=4
GRANULARITY = 16
stats = {'discarded_less_measures': 0, 
			'discarded_time_signature': 0, 
			'discarded_note_range': 0,
			'discarded_more_parts': 0,
			'discarded_high_granularity': 0,
			'total_cut_scores': 0,
			'total_scores': 0,
			'total_measures': 0}

def midi_to_note(midi_val):
	octave = int((midi_val-12) / 12)
	notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	note = notes[midi_val % 12]
	return note + str(octave)

assert midi_to_note(108) == 'C8'
assert midi_to_note(21) == 'A0'

# TODO: add check for number of beats actually in measure
# TODO: add key signature to encoding
# TODO: add new vs continued note distinction in encoding
# TODO: add support for pieces with pickups
# TODO: what happens when time signature denominator != 4
# TODO: remove dependency on midi_to_note function
def encode_score(score):
	assert len(score.parts) == 2 # discarded num parts
	score = score.flattenParts()
	score_X = []
	start_ind = 1
	cut_score = score.measures(start_ind, start_ind+MEASURES_PER_CUT-1)
	measures = cut_score.getElementsByClass('Measure')
	while len(measures) == MEASURES_PER_CUT:
		for ts in cut_score.getTimeSignatures():
			assert ts.ratioString == TIME_SIGNATURE.ratioString # discarded time signature
			print(ts.ratioString)
		
		x = np.zeros((int(MEASURES_PER_CUT*GRANULARITY*TIME_SIGNATURE.beatCount/4), NOTE_RANGE))
		for measure in measures:
			notes = measure.getElementsByClass(music21.note.Note)
			chords = measure.getElementsByClass(music21.chord.Chord)
			for note in notes:
				if note.quarterLength < 4.0 / GRANULARITY:
					continue # discarded high granularity
				pitch = note.pitches[0].midi
				if pitch < MIN_PITCH or pitch > MAX_PITCH:
					continue # discarded pitch out of range
				ind = (measure.measureNumber-1) % MEASURES_PER_CUT
				ind *= int(GRANULARITY*TIME_SIGNATURE.beatCount/4)
				ind += int(note.offset * GRANULARITY/4) # TODO: does offset work here?
				for i in range(int(note.quarterLength * GRANULARITY / 4)):
					x[ind+i][pitch-MIN_PITCH] = 1
			for chord in chords:
				if chord.quarterLength < 4.0 / GRANULARITY:
					continue # discarded high granularity
				for pitch in chord.pitches:
					if pitch.midi < MIN_PITCH or pitch.midi > MAX_PITCH:
						continue # discarded pitch out of range
					ind = (measure.measureNumber-1) % MEASURES_PER_CUT
					ind *= int(GRANULARITY*TIME_SIGNATURE.beatCount / 4)
					ind += int(chord.offset * GRANULARITY/4) # TODO: does offset work here?
					for i in range(int(chord.quarterLength * GRANULARITY / 4)):
						x[ind+i][pitch.midi-MIN_PITCH] = 1
		score_X.append(x)
		start_ind += MEASURES_PER_CUT
		cut_score = score.measures(start_ind, start_ind+MEASURES_PER_CUT-1)
		measures = cut_score.getElementsByClass('Measure')
	return score_X

def decode_score(encoding):
	print(len(encoding))
	assert len(encoding) == MEASURES_PER_CUT * GRANULARITY * TIME_SIGNATURE.beatCount / 4
	score = music21.stream.Stream()
	score.timeSignature = TIME_SIGNATURE
	measure_ind = 0
	while measure_ind < MEASURES_PER_CUT:
		start_beat = int(measure_ind * GRANULARITY * TIME_SIGNATURE.beatCount / 4)
		end_beat = int((measure_ind + 1) * GRANULARITY * TIME_SIGNATURE.beatCount / 4)
		measure = music21.stream.Measure()
		for beat_ind in range(start_beat, end_beat):
			played_pitches = np.nonzero(encoding[beat_ind])[0]
			if len(played_pitches) == 0:
				measure.append(music21.note.Rest(quarterLength=4.0/GRANULARITY))
			else:
				played_notes = [midi_to_note(int(pitch+MIN_PITCH)) for pitch in played_pitches]
				chord = music21.chord.Chord(played_notes, quarterLength=4.0/GRANULARITY)
				measure.append(chord)
		score.append(measure)
		measure_ind += 1
	score.show()
	return score

X = []
Y = []
for song_path in corpus:
	try: 
		score = music21.converter.parse(song_path)
		X_score = encode_score(score)
		X.extend(X_score)
		Y.append('mozart')
		print("Encoded ", song_path)
	except AssertionError:
		pass
	except ZeroDivisionError:
		pass

decode_score(X[0])