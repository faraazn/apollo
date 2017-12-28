import os
import glob
import music21
from music21.note import Note
from music21.chord import Chord
from music21.meter import TimeSignature
from music21.key import KeySignature
from music21.note import Rest
from music21.stream import Measure
from music21.stream import Stream
import numpy as np
import matplotlib.pyplot as plt

CORPUS_DIR = '/Users/faraaz/workspace/apollo/data/classical-musicxml/'
COMPOSERS = ['mozart']

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
cumulative_score_stats = {
	'composer': {},
	'period': {},
	'num_parts': {},
	'has_pickup': {},
	'num_measures': {},
	'consistent_measures': {},
	'min_note': {},
	'max_note': {},
	'granularity': {},
	'power_2_notes': {},
	'time_signatures': {},
	'key_signatures': {},
	'consistent_key': {},
	'consistent_time': {},
	'consistent_parts': {}
}
score_to_stats = {}

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
		# use recurse on notes?
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
	
def prune_dataset(score_names, time_signatures=set(), pickups=False, parts=set(), note_range=[], \
		num_measures=0, key_signatures=set(), granularity=0, consistent_measures=False, \
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
	
	for score_name in score_names:
		assert score_name in score_to_stats
		score_stats = score_to_stats[score_name]
	
		discarded = False
		
		if parts and score_stats['num_parts'] not in parts:
			discarded = True
			pruning_stats['discarded_num_parts'].add(score_name)
		if time_signatures and not score_stats['time_signatures'].issubset(time_signatures):
			discarded = True
			pruning_stats['discarded_time_signature'].add(score_name)
		if key_signatures and not score_stats['key_signatures'].issubset(key_signatures):
			discarded = True
			pruning_stats['discarded_key_signature'].add(score_name)
		if pickups and score_stats['has_pickup']:
			discarded = True
			pruning_stats['discarded_has_pickup'].add(score_name)
		if num_measures and score_stats['num_measures'] < num_measures:
			discarded = True
			pruning_stats['discarded_num_measures'].add(score_name)
		if note_range and score_stats['min_note'] >= note_range[0] and score_stats['max_note'] <= note_range[1]:
			discarded = True
			pruning_stats['discarded_note_range'].add(score_name)
		if consistent_measures and not score_stats['consistent_measures']:
			discarded = True
			pruning_stats['discarded_consistent_measures'].add(score_name)
		if granularity and score_stats['granularity'] > granularity:
			discarded = True
			pruning_stats['discarded_granularity'].add(score_name)
		if consistent_time and not score_stats['consistent_time']:
			discarded = True
			pruning_stats['discarded_consistent_time'].add(score_name)
		if consistent_key and not score_stats['consistent_key']:
			discarded = True
			pruning_stats['discarded_consistent_key'].add(score_name)
		if consistent_parts and not score_stats['consistent_parts']:
			discarded = True
			pruning_stats['discarded_consistent_parts'].add(score_name)
		
		if not discarded:
			pruned_dataset.append(score_name)
	
	return pruned_dataset

def get_score_stats(score_name, score, composer, period):
	if score_name in score_to_stats:
		return score_to_stats[score_name]
	
	score_stats = {}
	score_stats['composer'] = composer
	score_stats['period'] = period
	score_stats['num_parts'] = len(score.parts)
	score_stats['has_pickup'] = score.parts[0].measure(1) is not score.parts[0].getElementsByClass(Measure)[0]
	score_stats['num_measures'] = len(score.parts[0].getElementsByClass(Measure))
	score_stats['consistent_measures'] = not np.any(np.diff(np.diff(sorted(score.parts[0].measureOffsetMap().keys()))))
	
	min_note = None
	max_note = None
	granularity = None
	power_2_notes = True
	for note in score.recurse(classFilter=music21.note.GeneralNote):
		if note.isChord or note.isNote:
			for pitch in note.pitches:
				if min_note == None or pitch.midi < min_note:
					min_note = pitch.midi
				if max_note == None or pitch.midi > max_note:
					max_note = pitch.midi
		if note.quarterLength != 0:
			note_gran = int(1.0 / note.quarterLength)
			if granularity == None or note_gran > granularity:
				granularity = note_gran
			if not (note_gran != 0 and ((note_gran & (note_gran - 1)) == 0)):
				power_2_notes = False
	score_stats['min_note'] = min_note
	score_stats['max_note'] = max_note
	score_stats['granularity'] = granularity
	score_stats['power_2_notes'] = power_2_notes
	
	score_stats['time_signatures'] = frozenset(ts.ratioString for ts in score.recurse(classFilter=TimeSignature))
	score_stats['key_signatures'] = frozenset(ks.getScale('major').name for ks in score.recurse(classFilter=KeySignature))
	
	score_stats['consistent_key'] = len(score_stats['key_signatures']) == 1
	score_stats['consistent_time'] = len(score_stats['time_signatures']) == 1
	score_stats['consistent_parts'] = True # TODO: implement this
	
	return score_stats

def plot_statistic(stat):
	plt.bar(range(len(stat)), [len(val) for val in stat.values()], align='center')
	plt.xticks(range(len(stat)), list(stat.keys()))
	plt.show()

X_score = []
X_score_name = []
Y_composer = []
Y_era = []

for composer in COMPOSERS:
	score_names = [os.path.basename(path) for path in glob.glob(CORPUS_DIR+composer+"/*.xml")]
	for score_name in score_names:
		try:
			score = music21.converter.parse(CORPUS_DIR+composer+"/"+score_name)
			score_stats = get_score_stats(score_name, score, composer, 'classical')
			X_score.append(score)
			X_score_name.append(score_name)
			Y_composer.append(composer)
			Y_era.append('classical')
			for key in score_stats:
				if score_stats[key] in cumulative_score_stats[key]:
					cumulative_score_stats[key][score_stats[key]].add(score_name)
				else:
					cumulative_score_stats[key][score_stats[key]] = set([score_name])
			score_to_stats[score_name] = score_stats
		except ZeroDivisionError:
			pruning_stats['discarded_parse_error'].add(score_name)

X_score_name_pruned = prune_dataset(X_score_name, time_signatures=set(['3/4', '6/8']))
for key in pruning_stats:
	print(key + ": " + str(len(pruning_stats[key])))

plot_statistic(cumulative_score_stats['time_signatures'])