from time import time
import music21
from music21.note import Note
from music21.note import GeneralNote
from music21.chord import Chord
from music21.meter import TimeSignature
from music21.key import KeySignature
from music21.note import Rest
from music21.stream import Measure
from music21.stream import Stream
import numpy as np
import pickle

TASK_DIR = '/Users/faraaz/workspace/apollo/task_1/data/'

CORPUS_DIR = '/Users/faraaz/workspace/apollo/data/xml/'
COMPOSERS = ['bach', 'beethoven']
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
def midi_to_note(midi_val):
	octave = int((midi_val-12) / 12)
	notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	note = notes[midi_val % 12]
	return note + str(octave)

def encode_score(score, num_measures, steps_per_cut, image=False):
	if image:
		X_score = np.zeros((steps_per_cut, NOTE_RANGE, 1))
	else:
		X_score = np.zeros((steps_per_cut, NOTE_RANGE))
	steps_per_measure = steps_per_cut / num_measures
	for note in score.recurse(classFilter=GeneralNote):
		if (note.isChord or note.isNote) and note.quarterLength % (4.0 / GRANULARITY) == 0 :
			for pitch in note.pitches:
				ind = (note.measureNumber - 1) % num_measures
				ind *= steps_per_measure
				ind += note.offset * GRANULARITY / 4.0
				ind = int(ind)
				for i in range(int(note.quarterLength * GRANULARITY / 4.0)):
					if image:
						X_score[ind+i][pitch.midi-MIN_PITCH][0] = 1
					else:
						X_score[ind+i][pitch.midi-MIN_PITCH] = 1
	return X_score

def decode_score(encoding, num_measures, ts, image=False):
	score = Stream()
	score.timeSignature = TimeSignature(ts)
	steps_per_measure = len(encoding) / num_measures
	measure_ind = 0
	while measure_ind < num_measures:
		start_beat = int(measure_ind * steps_per_measure)
		end_beat = int((measure_ind + 1) * steps_per_measure)
		measure = Measure()
		for beat_ind in range(start_beat, end_beat):
			if image:
				played_pitches = np.nonzero(encoding[beat_ind])[0]
			else:
				played_pitches = np.nonzero(encoding[beat_ind])
			if len(played_pitches) == 0:
				measure.append(Rest(quarterLength=4.0/GRANULARITY))
			else:
				played_notes = [midi_to_note(int(pitch+MIN_PITCH)) for pitch in played_pitches]
				chord = Chord(played_notes, quarterLength=4.0/GRANULARITY)
				measure.append(chord)
		score.append(measure)
		measure_ind += 1
	return score

print("Loading sets...")
ts = time()
train_set = pickle.load(open('train_0.p', 'rb'))
valid_set = pickle.load(open('valid_0.p', 'rb'))
test_set = pickle.load(open('test_0.p', 'rb'))
score_to_stats = pickle.load(open('score_to_stats_0.p', 'rb'))
print('loading time {}s'.format(time() - ts))

print("Encoding dataset...")
X_score = []
X_score_name = []
Y_composer = []
ts = time()
for partition in [valid_set, train_set, test_set]:
	total = len(partition)
	for i, score_name in enumerate(partition):
		if i % 100 == 0:
			print(i, '/', total, ':', score_name)
		composer = score_to_stats[score_name]['composer']
		score = music21.converter.parse(TASK_DIR+composer+'/'+score_name+'.xml')
		encoded_score = encode_score(score, score_to_stats[score_name]['num_measures'], 192)
		X_score.append(encoded_score)
		X_score_name.append(score_name)
		if composer == 'bach':
			Y_composer.append(1)
		else:
			Y_composer.append(0)
X_score = np.array(X_score)
print(X_score.shape)
np.save("X", X_score)
np.save("score_names", X_score_name)
np.save("Y", Y_composer)
print(len(Y_composer))
print('encoding time {}s'.format(time() - ts))

print("Decoding dataset...")
X_score = np.load("X.npy")
X_score_name = np.load("score_names.npy")
Y_comppser = np.load("Y.npy")
total = len(X_score)
ts = time()
for i, score in enumerate(X_score):
	if i % 100 == 0:
		print(i, '/', total, ':', score_name)
	score_name = X_score_name[i]
	composer = Y_composer[i]
	ts = list(score_to_stats[score_name]['time_signatures'])[0]
	num_measures = score_to_stats[score_name]['num_measures']
	decoded_score = decode_score(score, num_measures, ts)
	decoded_score.show()
	break
# print('decoding time {}s'.format(time() - ts))

print("Done.")