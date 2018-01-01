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
import numpy as np
import matplotlib.pyplot as plt
import logging
from time import time
# import cPickle
from queue import Queue
from threading import Thread

CORPUS_DIR = '/Users/faraaz/workspace/apollo/data/midi/'
COMPOSERS = ['bach']
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
TIME_SIGNATURE = TimeSignature('4/4')
GRANULARITY = 16
STEPS_PER_MEASURE = GRANULARITY*TIME_SIGNATURE.beatCount*TIME_SIGNATURE.beatDuration.quarterLength/4.0
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
	'discarded_%_indivisible': set()
	}
cumulative_score_stats = {}
score_to_stats = {}

def midi_to_note(midi_val):
	octave = int((midi_val-12) / 12)
	notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	note = notes[midi_val % 12]
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
		'1%+_indivisible': {}
	}

def get_cut_score(score, measures_per_cut):
	X_cut_score = []
	start_ind = 1
	cut_score = score.measures(start_ind, start_ind+measures_per_cut-1)
	while len(cut_score.parts[0].getElementsByClass(Measure)) == measures_per_cut:
		X_cut_score.append(cut_score)
		start_ind += measures_per_cut
		cut_score = score.measures(start_ind, start_ind+measures_per_cut-1)
	
	return X_cut_score

# TODO: add key signature to encoding
# TODO: add new vs continued note distinction in encoding
# TODO: remove dependency on midi_to_note function
def encode_score(score):
	X_score = np.zeros((int(MEASURES_PER_CUT * STEPS_PER_MEASURE), NOTE_RANGE))
	for note in score.recurse(classFilter=GeneralNote):
		if (note.isChord or note.isNote) and note.quarterLength % (4.0 / GRANULARITY) == 0:
			for pitch in note.pitches:
				ind = (note.measureNumber - 1) % MEASURES_PER_CUT
				ind *= STEPS_PER_MEASURE
				ind += note.offset * GRANULARITY / 4.0
				ind = int(ind)
				for i in range(int(note.quarterLength * GRANULARITY / 4.0)):
					X_score[ind+i][pitch.midi-MIN_PITCH] = 1
	return X_score

def decode_score(encoding):
	assert len(encoding) == MEASURES_PER_CUT * STEPS_PER_MEASURE
	score = Stream()
	score.timeSignature = TIME_SIGNATURE
	measure_ind = 0
	while measure_ind < MEASURES_PER_CUT:
		start_beat = int(measure_ind * STEPS_PER_MEASURE)
		end_beat = int((measure_ind + 1) * STEPS_PER_MEASURE)
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
		num_measures=0, key_signatures=set(), granularity=0, consistent_measures=False, consistent_time=False, \
		consistent_key=False, consistent_parts=False, percent_indivisible=0.0, has_key_signature=False):
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
		if note_range and note_range[0] and note_range[1] and (score_stats['min_note'] < note_range[0] or score_stats['max_note'] > note_range[1]):
			discarded = True
			pruning_stats['discarded_note_range'].add(score_name)
		if consistent_measures and not score_stats['consistent_measures']:
			discarded = True
			pruning_stats['discarded_consistent_measures'].add(score_name)
		if granularity and score_stats['granularity'] > granularity:
			discarded = True
			pruning_stats['discarded_granularity'].add(score_name)
		if percent_indivisible and score_stats['1%+_indivisible']:
			discarded = True
			pruning_stats['discarded_%_indivisible'].add(score_name)
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
	# Tested
	score_stats['composer'] = composer
	# Tested
	score_stats['period'] = period
	
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
	score_stats['1%+_indivisible'] = indivisible_notes / total_notes > 0.01
	
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

class DownloadWorker(Thread):
	def __init__(self, queue):
		Thread.__init__(self)
		self.queue = queue
	def run(self):
		while True:
			# Get the work from the queue and expand the tuple
			ind, score_name, composer = self.queue.get()
			global X_score_composer
			global COMPOSER_TO_ERA
			global CORPUS_DIR
			midi_path = CORPUS_DIR+composer+"/"+score_name
			try:
				mf = MidiFile()
				mf.open(midi_path)
				mf.read()
				mf.close()
				score = music21.midi.translate.midiFileToStream(mf)
				X_score_composer[ind] = score
				score_stats = get_score_stats(score_name, score, composer, COMPOSER_TO_ERA[composer])
				for key in score_stats:
					if score_stats[key] in cumulative_score_stats[key]:
						cumulative_score_stats[key][score_stats[key]].add(score_name)
					else:
						cumulative_score_stats[key][score_stats[key]] = set([score_name])
					score_to_stats[score_name] = score_stats
			except ZeroDivisionError:
				pruning_stats['discarded_parse_error'].add(score_name)
			print("finished", score_name)
			self.queue.task_done()


print("Loading dataset...")
reset_cumulative_stats()
X_score = []
X_score_composer = [] # threadsafe object
X_score_name = []
Y_composer = []
Y_era = []
for composer in COMPOSERS:
	print("Loading", composer)
# 	score_names = [os.path.basename(path) for path in glob.glob(CORPUS_DIR+composer+"/*.xml")]
	score_names = [os.path.basename(path) for path in glob.glob(CORPUS_DIR+composer+"/*.mid")][:5]
	total = len(score_names)
	for i, score_name in enumerate(score_names):
		print(i, score_name)
		try:
			ts = time()
			mf = MidiFile()
			mf.open(CORPUS_DIR+composer+"/"+score_name)
			mf.read()
			mf.close()
			print('reading file {}s'.format(time() - ts))
			ts = time()
# 			score = music21.converter.parse(CORPUS_DIR+composer+"/"+score_name)
			score = music21.midi.translate.midiFileToStream(mf)
			print('converting midi {}s'.format(time() - ts))
			ts = time()
			score_stats = get_score_stats(score_name, score, composer, COMPOSER_TO_ERA[composer])
			print('score_stats {}s'.format(time() - ts))
			X_score.append(score)
			X_score_name.append(score_name)
			Y_composer.append(composer)
			Y_era.append(COMPOSER_TO_ERA[composer])
			for key in score_stats:
				if score_stats[key] in cumulative_score_stats[key]:
					cumulative_score_stats[key][score_stats[key]].add(score_name)
				else:
					cumulative_score_stats[key][score_stats[key]] = set([score_name])
			score_to_stats[score_name] = score_stats
		except ZeroDivisionError:
			pruning_stats['discarded_parse_error'].add(score_name)
# ts = time()
# for composer in COMPOSERS:
# 	print("Loading", composer)
# 	score_names = [os.path.basename(path) for path in glob.glob(CORPUS_DIR+composer+"/*.mid")][:50]
# 	total = len(score_names)
# 	X_score_composer = [0]*total
# 	# Create a queue to communicate with the worker threads
# 	queue = Queue()
# 	# Create 8 worker threads
# 	for x in range(8):
# 		worker = DownloadWorker(queue)
# 		# Setting daemon to True will let the main thread exit even though the workers are blocking
# 		worker.daemon = True
# 		worker.start()
# 	# Put the tasks into the queue as a tuple
# 	for i, score_name in enumerate(score_names):
# 		queue.put((i, score_name, composer))
# 	# Causes the main thread to wait for the queue to finish processing all the tasks
# 	queue.join()
# 	X_score.extend(X_score_composer)
# 	X_score_name.extend(score_names)
# 	Y_composer.extend([composer]*total)
# 	Y_era.extend(COMPOSER_TO_ERA[composer]*total)
# print('Took {}s'.format(time() - ts))

print("Partitioning dataset...")
X_cut_score = []
X_cut_score_name = []
Y_cut_composer = []
Y_cut_era = []
total = len(X_score_name)
for i, score_name in enumerate(X_score_name):
	print(i, score_name)
	score = X_score[i]
	composer = Y_composer[i]
	era = Y_era[i]
	cut_scores = get_cut_score(score, MEASURES_PER_CUT)
	X_cut_score.extend(cut_scores)
	X_cut_score_name.extend([score_name+"-"+str(num) for num in range(len(cut_scores))])
	Y_cut_composer.extend([composer for _ in range(len(cut_scores))])
	Y_cut_era.extend([era for _ in range(len(cut_scores))])
del X_score
del X_score_name
del Y_composer
del Y_era

print("Extracting dataset info...")
reset_cumulative_stats()
for i, score_name in enumerate(X_cut_score_name):
	print(i, score_name)
	score = X_cut_score[i]
	composer = Y_cut_composer[i]
	era = Y_cut_era[i]
	score_stats = get_score_stats(score_name, score, composer, era)
	for key in score_stats:
		if score_stats[key] in cumulative_score_stats[key]:
			cumulative_score_stats[key][score_stats[key]].add(score_name)
		else:
			cumulative_score_stats[key][score_stats[key]] = set([score_name])
	score_to_stats[score_name] = score_stats

for stat in cumulative_score_stats:
	plot_statistic(cumulative_score_stats[stat], stat)

print("Pruning dataset...")
reset_cumulative_stats()
X_pruned_score_name = prune_dataset(X_cut_score_name, \
		time_signatures=set([TIME_SIGNATURE.ratioString]), \
		pickups=True, \
		parts=set([2]), \
		note_range=[MIN_PITCH, MAX_PITCH], \
		num_measures=MEASURES_PER_CUT, \
		consistent_measures=True, \
		consistent_time=True, \
		consistent_key=False, \
		consistent_parts=True, \
		percent_indivisible=True, \
		has_key_signature=True)
X_pruned_score = []
Y_pruned_composer = []
Y_pruned_era = []
for i, score_name in enumerate(X_pruned_score_name):
	print(i)
	ind = X_cut_score_name.index(score_name)
	score = X_cut_score[ind]
	composer = Y_cut_composer[ind]
	era = Y_cut_era[ind]
	score_stats = get_score_stats(score_name, score, composer, era)
	for key in score_stats:
		if score_stats[key] in cumulative_score_stats[key]:
			cumulative_score_stats[key][score_stats[key]].add(score_name)
		else:
			cumulative_score_stats[key][score_stats[key]] = set([score_name])
	score_to_stats[score_name] = score_stats
	X_pruned_score.append(score)
	Y_pruned_composer.append(composer)
	Y_pruned_era.append(era)
del X_cut_score
del X_cut_score_name
del Y_cut_composer
del Y_cut_era

for stat in cumulative_score_stats:
	plot_statistic(cumulative_score_stats[stat], stat)

for stat in pruning_stats:
	print(stat, ":", len(pruning_stats[stat]))

# for val in cumulative_score_stats['key_signatures']:
# 	print(val)
# 	for score_name in cumulative_score_stats['key_signatures'][val]:
# 		score = X_cut_score[X_cut_score_name.index(score_name)]
# 		print(score_name)
# 		score.show()

print("Encoding dataset...")
X_encoded_score = []
X_encoded_score_name = []
Y_encoded_composer = []
Y_encoded_era = []
for i, score_name in enumerate(X_pruned_score_name):
	print(i)
	score = X_pruned_score[i]
	composer = Y_pruned_composer[i]
	era = Y_pruned_era[i]
	encoded_score = encode_score(score)
	X_encoded_score.append(encoded_score)
	X_encoded_score_name.append(score_name)
	Y_encoded_composer.append(composer)
	Y_encoded_era.append(era)
X_encoded_score = np.array(X_encoded_score)
print(X_encoded_score.shape)

print("Decoding dataset...")
for i, score in enumerate(X_encoded_score):
	print(i)
	decoded_score = decode_score(X_encoded_score[i])
	decoded_score.show()

print("Done.")