import random
import pickle
from time import time
from music21.note import Note
from music21.meter import TimeSignature

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
	'discarded_%_divisible': set(),
	'discarded_num_steps': set()
	}

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJORS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'B-', 'E-', 'A-', 'D-', 'G-', 'C-']

def valid_score(score_name, time_signatures=set(), pickups=False, parts=set(), note_range=[], num_measures=0, \
	key_signatures=set(), granularity=0, consistent_measures=False, consistent_time=False, consistent_key=False, \
	consistent_parts=False, percent_indivisible=0.0, has_key_signature=False, num_steps=0):
		
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
	if percent_indivisible and score_stats['1%+_divisible']:
		discarded = True
		pruning_stats['discarded_%_divisible'].add(score_name)
	if consistent_time and not score_stats['consistent_time']:
		discarded = True
		pruning_stats['discarded_consistent_time'].add(score_name)
	if consistent_key and not score_stats['consistent_key']:
		discarded = True
		pruning_stats['discarded_consistent_key'].add(score_name)
	if consistent_parts and not score_stats['consistent_parts']:
		discarded = True
		pruning_stats['discarded_consistent_parts'].add(score_name)
	if num_steps:
		if len(score_stats['time_signatures']) == 1:
			ts = TimeSignature(list(score_stats['time_signatures'])[0])
			if num_steps != score_stats['num_measures']*(GRANULARITY*ts.beatCount*ts.beatDuration.quarterLength/4.0):
				discarded = True
				pruning_stats['discarded_num_steps'].add(score_name)
		else:
			discarded = True
			pruning_stats['discarded_num_steps'].add(score_name)
	return discarded

print("Loading stats...")
ts = time()
cumulative_score_stats = pickle.load(open('cumulative_score_stats_0.p', 'rb'))
score_to_stats = pickle.load(open('score_to_stats_0.p', 'rb'))
print('loading time {}s'.format(time() - ts))

print("Splitting dataset...")
ts = time()
train = set()
train_scores = set()
valid = set()
valid_scores = set()
test = set()
test_scores = set()
for score_name in score_to_stats:
	if not valid_score(score_name, \
			parts=set([1, 2, 3, 4]), \
			note_range=[MIN_PITCH, MAX_PITCH], \
			percent_indivisible=True, \
			num_steps=STEPS_PER_CUT):
		continue
	# keep augmentations or pieces from same score together
	if '-' not in score_name:
		continue
	og_score_name = score_name[:score_name.index('-')]
	if og_score_name in train_scores:
		train.add(score_name)
	elif og_score_name in valid_scores:
		valid.add(score_name)
	elif og_score_name in test_scores:
		test.add(score_name)
	else:
		n = random.randint(0,9)
		if n < 7: # 70% train set
			train_scores.add(og_score_name)
			train.add(score_name)
		elif n == 7: # 10% valid set
			valid_scores.add(og_score_name)
			valid.add(score_name)
		else: # 20% test set
			test_scores.add(og_score_name)
			test.add(score_name)
print("train:", len(train), len(train_scores))
print("valid:", len(valid), len(valid_scores))
print("test:", len(test), len(test_scores))
print('splitting time {}s'.format(time() - ts))

for stat in pruning_stats:
	print(stat, ":", len(pruning_stats[stat]))

print("Pickling sets...")
ts = time()
pickle.dump(train, open('train.p', 'wb'))
pickle.dump(valid, open('valid.p', 'wb'))
pickle.dump(test, open('test.p', 'wb'))
print('pickling time {}s'.format(time() - ts))