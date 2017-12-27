import os
import music21
from music21.note import Note
from music21.chord import Chord
from music21.meter import TimeSignature
from music21.note import Rest
from music21.stream import Measure
from music21.stream import Stream
import numpy as np

CORPUS_DIR = '/Users/faraaz/workspace/apollo/data/classical-musicxml/mozart/'
corpus = ['m5937_8.xml', 'm15791_5.xml', 'm15715_5.xml', 'm15714_5.xml', 'm17146_3.xml']

MEASURES_PER_CUT = 32
MAX_NOTE = Note('C8')
MIN_NOTE = Note('A0')
MAX_PITCH = MAX_NOTE.pitches[0].midi
MIN_PITCH = MIN_NOTE.pitches[0].midi
assert MAX_PITCH == 108
assert MIN_PITCH == 21
NOTE_RANGE = int(MAX_PITCH - MIN_PITCH + 1)
TIME_SIGNATURE = TimeSignature('3/4')
GRANULARITY = 16
pruning_stats = {
	'discarded_num_measures': set(),
	'discarded_time_signature': set(),
	'discarded_key_signature': set(),
	'discarded_note_range': set(),
	'discarded_num_parts': set(),
	'discarded_granularity': set(),
	'discarded_has_pickup': set(),
	'discarded_parse_error': set(),
	'discarded_consistent_key': set(),
	'discarded_consistent_time': set(),
	'discarded_consistent_measures': set()
	}

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
	measures = cut_score.getElementsByClass(Measure)
	while len(measures) == MEASURES_PER_CUT:
		for ts in cut_score.getTimeSignatures():
			assert ts.ratioString == TIME_SIGNATURE.ratioString # discarded time signature
			print(ts.ratioString)
		
		x = np.zeros((int(MEASURES_PER_CUT*GRANULARITY*TIME_SIGNATURE.beatCount*TIME_SIGNATURE.beatDuration.quarterLength/4), NOTE_RANGE))
		for measure in measures: # TODO: use offsetMap instead?
			notes = measure.getElementsByClass(Note)
			chords = measure.getElementsByClass(Chord)
			for note in notes:
				if note.quarterLength < 4.0 / GRANULARITY:
					continue # discarded high granularity
				pitch = note.pitches[0].midi
				if pitch < MIN_PITCH or pitch > MAX_PITCH:
					continue # discarded pitch out of range
				ind = (measure.measureNumber-1) % MEASURES_PER_CUT
				ind *= int(GRANULARITY*TIME_SIGNATURE.beatCount*TIME_SIGNATURE.beatDuration.quarterLength/4)
				ind += int(note.offset * GRANULARITY/4) 
				for i in range(int(note.quarterLength * GRANULARITY / 4)):
					x[ind+i][pitch-MIN_PITCH] = 1
			for chord in chords:
				if chord.quarterLength < 4.0 / GRANULARITY:
					continue # discarded high granularity
				for pitch in chord.pitches:
					if pitch.midi < MIN_PITCH or pitch.midi > MAX_PITCH:
						continue # discarded pitch out of range
					ind = (measure.measureNumber-1) % MEASURES_PER_CUT
					ind *= int(GRANULARITY*TIME_SIGNATURE.beatCount * TIME_SIGNATURE.beatDuration.quarterLength / 4)
					ind += int(chord.offset * GRANULARITY/4) 
					for i in range(int(chord.quarterLength * GRANULARITY / 4)):
						x[ind+i][pitch.midi-MIN_PITCH] = 1
		score_X.append(x)
		start_ind += MEASURES_PER_CUT
		cut_score = score.measures(start_ind, start_ind+MEASURES_PER_CUT-1)
		measures = cut_score.getElementsByClass(Measure)
	return score_X

def decode_score(encoding):
	print(len(encoding))
	assert len(encoding) == MEASURES_PER_CUT * GRANULARITY * TIME_SIGNATURE.beatCount * TIME_SIGNATURE.beatDuration.quarterLength / 4
	score = Stream()
	score.timeSignature = TIME_SIGNATURE
	measure_ind = 0
	while measure_ind < MEASURES_PER_CUT:
		start_beat = int(measure_ind * GRANULARITY * TIME_SIGNATURE.beatCount * TIME_SIGNATURE.beatDuration.quarterLength / 4)
		end_beat = int((measure_ind + 1) * GRANULARITY * TIME_SIGNATURE.beatCount * TIME_SIGNATURE.beatDuration.quarterLength / 4)
		measure = Measure()
		for beat_ind in range(start_beat, end_beat):
			played_pitches = np.nonzero(encoding[beat_ind])[0]
			if len(played_pitches) == 0:
				measure.append(Rest(quarterLength=4.0/GRANULARITY))
			else:
				played_notes = [midi_to_note(int(pitch+MIN_PITCH)) for pitch in played_pitches]
				chord = Chord(played_notes, quarterLength=4.0/GRANULARITY)
				measure.append(chord)
		score.append(measure)
		measure_ind += 1
	score.show()
	return score
	
def prune_dataset(dataset, time_signatures=[], pickups=False, parts=[], note_range=[], \
		num_measures=0, key_signatures=[], granularity=0, consistent_measures=False, \
		consistent_time=False, consistent_key=False, consistent_parts=False):
	assert isinstance(num_measures, int) and num_measures >= 0
	assert isinstance(granularity, int) and granularity >= 0
	assert len(note_range) == 0 or (len(note_range) == 2 and note_range[0] <= note_range[1])
	for bound in note_range:
		assert isinstance(bound, int) 
	for p in parts:
		assert isinstance(p, int) and p >= 1
	for ts in time_signatures:
		assert isinstance(ts, str)
	for ks in key_signatures:
		assert isinstance(ks, str)
	
	pruned_dataset = []
	
	for score_name, score in dataset:
		discarded = False
		
		# Tested
		if parts and len(score.parts) not in parts:
			discarded = True
			pruning_stats['discarded_num_parts'].add(score_name)
		# Tested
		if time_signatures:
			for part in score.parts:
				if score_name in pruning_stats['discarded_time_signature']:
					break
				for ts in part.getTimeSignatures(): # what if contained in measures?
					if ts.ratioString not in time_signatures:
						discarded = True
						pruning_stats['discarded_time_signature'].add(score_name)
						break
		# Tested
		if key_signatures:
			for part in score.parts:
				if score_name in pruning_stats['discarded_key_signature']:
					break
				for measure in part.getElementsByClass(Measure): 
					if score_name in pruning_stats['discarded_key_signature']:
						break
					for ks in measure.getKeySignatures(): # what if contained in part?
						if ks.getScale('major').name not in key_signatures:
							discarded = True
							pruning_stats['discarded_key_signature'].add(score_name)
							break
		# Tested
		if pickups:
			for part in score.parts: 
				if part.measure(1) is not part.getElementsByClass(Measure)[0]:
					discarded = True
					pruning_stats['discarded_has_pickup'].add(score_name)
					break
		# Tested
		if num_measures:
			for part in score.parts:
				if len(part.getElementsByClass(Measure)) < num_measures: 
					discarded = True
					pruning_stats['discarded_num_measures'].add(score_name)
					break
		# TODO: test
		if note_range: # use recurse?
			for note in score.getElementsByClass(Note):
				if note.pitches[0].midi < note_range[0] or note.pitches[0].midi > note_range[1]:
					discarded = True
					pruning_stats['discarded_note_range'].add(score_name)
					break
			for chord in score.getElementsByClass(Chord):
				if score_name in pruning_stats['discarded_note_range']:
					break
				for pitch in chord.pitches:
					if pitch.midi < note_range[0] or pitch.midi > note_range[1]:
						discarded = True
						pruning_stats['discarded_note_range'].add(score_name)
						break
		# Tested
		if consistent_measures:
			for part in score.parts:
				if score_name in pruning_stats['discarded_consistent_measures']:
					break
				offsets = sorted(part.measureOffsetMap().keys())
				offset = offsets[1] - offsets[0]
				for i in range(len(offsets)):
					if i != 0 and offsets[i] - offsets[i-1] != offset:
						discarded = True
						pruning_stats['discarded_consistent_measures'].add(score_name)
						break
		# Tested
		if granularity:
			for note in score.recurse(classFilter=Note):
				if note.quarterLength <= 4.0 / granularity and note.quarterLength != 0:
					discarded = True
					pruning_stats['discarded_granularity'].add(score_name)
					break
			for chord in score.recurse(classFilter=Chord):
				if score_name in pruning_stats['discarded_granularity']:
					break
				if chord.quarterLength <= 4.0 / granularity and chord.quarterLength != 0:
					discarded = True
					pruning_stats['discarded_granularity'].add(score_name)
					break
		# TODO: test
		if consistent_time and len(score.getTimeSignatures()) > 1:
			discarded = True
			pruning_stats['discarded_consistent_time'].add(score_name)
		# TODO: test
		if consistent_key and len(score.getKeySignatures()) > 1:
			discarded = True
			pruning_stats['discarded_consistent_key'].add(score_name)
		# TODO: test
		if consistent_parts:
			offsets = score.parts[0].measureOffsetMap().keys()
			ts = score.parts[0].getTimeSignatures()
			ks = score.parts[0].getKeySignatures()
			for part in score.parts:
				if part.measureOffsetMap().keys != offsets or part.getTimeSignatures() != ts or part.getKeySignatures() != ks:
					discarded = True
					pruning_stats['discarded_consistent_parts'].add(score_name)
					break
		
		if not discarded:
			pruned_dataset.append(score)
	
	return pruned_dataset

X = []
Y = []
dataset = []
for score_name in corpus:
	try: 
		score = music21.converter.parse(CORPUS_DIR+score_name)
		dataset.append((score_name, score))
# 		X_score = encode_score(score)
# 		X.extend(X_score)
# 		Y.append('mozart')
# 		print("Encoded ", song_path)
	except ZeroDivisionError:
		pruning_stats['discarded_parse_error'].add(score_name)

prune_dataset(dataset, granularity=32)

for key in pruning_stats:
	print(key + ": " + str(len(pruning_stats[key])))