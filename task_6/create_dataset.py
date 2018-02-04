import os
import glob
import music21
from music21.note import Note
from music21.note import GeneralNote
from music21.chord import Chord
from music21.meter import TimeSignature
from music21.key import KeySignature
from music21.note import Rest
from music21.stream import Measure
from music21.stream import Stream
from music21.midi import MidiFile
from music21.musicxml.m21ToXml import ScoreExporter
from music21.meter import TimeSignatureException
from music21.duration import DurationException
import numpy as np
import matplotlib.pyplot as plt
import logging
from time import time
from queue import Queue
from threading import Thread
import pickle

"""
Task 1
Dataset: 
  - 2 part piano solo works
Dataset augmentations:
  - Key transpose
Categories:
  - Bach vs Beethoven
Encodings:
  - Piano roll
"""
TASK_DIR = '/Users/faraaz/workspace/apollo/task_6/data/'

CORPUS_DIR = '/Users/faraaz/workspace/apollo/data/xml/'
COMPOSERS = ['bach', 'handel', 'beethoven', 'mozart', 'chopin', 'strauss']
COMPOSER_TO_ERA = {
	'bach': 'baroque',
	'handel': 'baroque',
	'beethoven': 'classical',
	'mozart': 'classical',
	'chopin': 'romantic',
	'strauss': 'romantic'
}

MEASURES_PER_CUT = 16
MAX_NOTE = Note('C8')
MIN_NOTE = Note('A0')
MAX_PITCH = MAX_NOTE.pitches[0].midi
MIN_PITCH = MIN_NOTE.pitches[0].midi
assert MAX_PITCH == 108
assert MIN_PITCH == 21
NOTE_RANGE = int(MAX_PITCH - MIN_PITCH + 1)
GRANULARITY = 16
STEPS_PER_CUT = 48*4
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
	'discarded_consistent_measures': set(),
	'discarded_%_divisible': set()
	}
cumulative_score_stats = {}
score_to_stats = {}

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJORS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'B-', 'E-', 'A-', 'D-', 'G-', 'C-']
def midi_to_note(midi_val):
	octave = int((midi_val-12) / 12)
	note = NOTES[midi_val % 12]
	return note + str(octave)

assert midi_to_note(108) == 'C8'
assert midi_to_note(21) == 'A0'

def reset_cumulative_stats():
	global cumulative_score_stats
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
		'divisible_notes': {},
		'time_signatures': {},
		'key_signatures': {},
		'consistent_key': {},
		'consistent_time': {},
		'consistent_parts': {},
		'1%+_divisible': {},
		'%_indivisible': {}
	}

def get_cut_score_nummeasures(score, measures_per_cut, start_i=1):
	X_cut_score = []
	start_ind = start_i
	cut_score = score.measures(start_ind, start_ind+measures_per_cut-1)
	while len(cut_score.parts[0].getElementsByClass(Measure)) == measures_per_cut:
		X_cut_score.append(cut_score)
		start_ind += measures_per_cut
		cut_score = score.measures(start_ind, start_ind+measures_per_cut-1)
	return X_cut_score

def get_cut_score_numsteps(score, steps_per_cut):
	X_cut_score = []
	start_ind = 1
	i = 1
	total_measures = len(score.parts[0].getElementsByClass(Measure))
	ts = score.parts[0].measure(1).timeSignature
	while i <= total_measures:
		cur_ts = score.parts[0].measure(i).timeSignature
		if (cur_ts and cur_ts.ratioString != ts.ratioString) or i == total_measures:
			try:
				cut_score = score.measures(start_ind, i-1)
				measures_per_cut = int(steps_per_cut/(GRANULARITY*ts.beatCount*ts.beatDuration.quarterLength/4.0))
				X_cut_score.extend(get_cut_score_nummeasures(cut_score, measures_per_cut, start_ind))
			except TimeSignatureException:
				print("invalid ts:", ts.ratioString)
			ts = cur_ts
			start_ind = i
		i += 1
	return X_cut_score

def get_score_stats(score_name, score, composer):
	if score_name in score_to_stats:
		return score_to_stats[score_name]
	
	score_stats = {}
	# Tested
	score_stats['composer'] = composer
	
	# Tested
	score_stats['num_parts'] = len(score.parts)
	# Tested
	score_stats['has_pickup'] = len(score.measure(0).parts[0].getElementsByClass(Measure)) == 1
	# Tested
	score_stats['num_measures'] = len(score.parts[0].getElementsByClass(Measure))
	# Tested
	score_stats['consistent_measures'] = not np.any(np.diff(np.diff(sorted(score.parts[0].measureOffsetMap().keys()))))
	
	min_note = None
	max_note = None
	granularity = None
	divisible_notes = True
	total_notes = 0
	indivisible_notes = 0
	for note in score.recurse(classFilter=GeneralNote):
		total_notes += 1
		if note.isChord or note.isNote:
			for pitch in note.pitches:
				if min_note == None or pitch.midi < min_note:
					min_note = pitch.midi
				if max_note == None or pitch.midi > max_note:
					max_note = pitch.midi
		if note.quarterLength != 0:
			note_gran = 1.0 / (0.25 * note.quarterLength)
			if granularity == None or note_gran > granularity:
				granularity = note_gran
			if note.quarterLength % (4.0 / GRANULARITY) != 0:
				indivisible_notes += 1
				divisible_notes = False
	# Tested
	score_stats['min_note'] = min_note
	# Tested
	score_stats['max_note'] = max_note
	# Tested
	score_stats['granularity'] = granularity
	# Tested
	score_stats['divisible_notes'] = divisible_notes
	# Tested
	score_stats['1%+_divisible'] = indivisible_notes / total_notes < 0.01
	# Tested
	score_stats['%_indivisible'] = indivisible_notes / total_notes
	
	# Tested
	score_stats['time_signatures'] = frozenset(ts.ratioString for ts in score.recurse(classFilter=TimeSignature))
	# Tested
	score_stats['key_signatures'] = frozenset(ks.getScale('major').name for ks in score.recurse(classFilter=KeySignature))
	
	# Tested
	score_stats['consistent_key'] = len(score_stats['key_signatures']) == 1
	# Tested
	score_stats['consistent_time'] = len(score_stats['time_signatures']) == 1
	# TODO: implement this
	score_stats['consistent_parts'] = True
	
	return score_stats

def plot_statistic(stat, title):
	plt.bar(range(len(stat)), [len(val) for val in stat.values()], align='center')
	plt.xticks(range(len(stat)), list(stat.keys()))
	plt.title(title)
	plt.show()

def augment_score_keys(score):
	augmented_scores = []
	augmented_scores.append(score)
	for k in range(1, 6):
		aug_score1 = score.transpose(k)
		aug_score2 = score.transpose(-k)
		augmented_scores.append(aug_score1)
		augmented_scores.append(aug_score2)
	return augmented_scores

print("Processing dataset...")
reset_cumulative_stats()
X_score = []
X_score_name = []
Y_composer = []
ogts = time()
ts = time()
for composer in COMPOSERS:
	print("Loading", composer)
	score_names = [os.path.basename(path)[:-4] for path in glob.glob(CORPUS_DIR+composer+"/*.xml")]
	total = len(score_names)
	for i, score_name in enumerate(score_names):
		if i % 10 == 0:
			print(i, '/', total, ':', score_name, ',', str(time()-ts))
			ts = time()
		try:
			# PARSE
			score = music21.converter.parse(CORPUS_DIR+composer+'/'+score_name+'.xml')
			score_stats = get_score_stats(score_name, score, composer)
			for key in score_stats:
				if score_stats[key] in cumulative_score_stats[key]:
					cumulative_score_stats[key][score_stats[key]].add(score_name)
				else:
					cumulative_score_stats[key][score_stats[key]] = set([score_name])
			score_to_stats[score_name] = score_stats

			# PARTITION
			cut_scores = get_cut_score_numsteps(score, STEPS_PER_CUT)
			for j in range(len(cut_scores)):
				# AUGMENT
				aug_scores = augment_score_keys(cut_scores[j])
				aug_score_names = [score_name+"-"+str(j)+"-"+str(num) for num in range(len(aug_scores))]
				for k, aug_score in enumerate(aug_scores):
					aug_score_name = aug_score_names[i]
					try:
						# SAVE
						aug_score.write('musicxml', TASK_DIR+composer+'/'+aug_score_name+'.xml')
						score_stats = get_score_stats(aug_score_name, aug_score, composer)
						for key in score_stats:
							if score_stats[key] in cumulative_score_stats[key]:
								cumulative_score_stats[key][score_stats[key]].add(aug_score_name)
							else:
								cumulative_score_stats[key][score_stats[key]] = set([aug_score_name])
						score_to_stats[aug_score_name] = score_stats
					except DurationException:
						print("unable to save:", score_name)

		except ZeroDivisionError:
			pruning_stats['discarded_parse_error'].add(score_name)
		break
print('total processing time {}s'.format(time() - ogts))

print("Pickling stats...")
ts = time()
pickle.dump(cumulative_score_stats, open('cumulative_score_stats.p', 'wb'))
pickle.dump(score_to_stats, open('score_to_stats.p', 'wb'))
print('pickling time {}s'.format(time() - ts))

for stat in cumulative_score_stats:
	plot_statistic(cumulative_score_stats[stat], stat)