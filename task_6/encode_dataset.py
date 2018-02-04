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
import midi

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
def midi_to_note(midi_val):
    octave = int((midi_val-12) / 12)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = notes[midi_val % 12]
    return note + str(octave)

def encode_score(score, num_measures, steps_per_cut):
    X_score = np.zeros((steps_per_cut, NOTE_RANGE, 3))
    steps_per_measure = steps_per_cut / num_measures
    for note in score.recurse(classFilter=GeneralNote):
        if (note.isChord or note.isNote) and note.quarterLength % (4.0 / GRANULARITY) == 0 :
            for pitch in note.pitches:
                ind = (note.measureNumber - 1) % num_measures
                ind *= steps_per_measure
                ind += note.offset * GRANULARITY / 4.0
                ind = int(ind)
                for i in range(int(note.quarterLength * GRANULARITY / 4.0)):
                    if i == 0:
                        X_score[ind+i][pitch.midi-MIN_PITCH][1] = 1
                    else:
                        X_score[ind+i][pitch.midi-MIN_PITCH][2] = 1
    return X_score

def decode_score(piece, name):
    lowerBound = 21
    upperBound = 109
    span = upperBound-lowerBound

    statematrix = np.argmax(piece, axis=2)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    tickscale = 55

    lastcmdtime = 0
    prevstate = [0 for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p == 0:
                if n == 1:
                    onNotes.append(i)
            elif p == 1:
                if n == 0:
                    offNotes.append(i)
                if n == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif p == 2:
                if n == 0:
                    offNotes.append(i)
                if n == 1:
                    offNotes.append(i)
                    onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

print("Loading sets...")
ts = time()
train_set = pickle.load(open('train_0.p', 'rb'))
score_to_stats = pickle.load(open('score_to_stats_0.p', 'rb'))
print('loading time {}s'.format(time() - ts))

print("Encoding dataset...")
X_score = []
X_score_name = []
Y_composer = []
total = len(train_set)
ts = time()
for i, score_name in enumerate(train_set):
    if i % 100 == 0:
        print(i, '/', total, ':', score_name)
    composer = score_to_stats[score_name]['composer']
    score = music21.converter.parse(TASK_DIR+composer+'/'+score_name+'.xml')
    encoded_score = encode_score(score, score_to_stats[score_name]['num_measures'], 192)
    X_score.append(encoded_score)
    X_score_name.append(score_name)
    Y_composer.append(COMPOSERS.index(composer))
X_score = np.array(X_score)
np.save("X", X_score)
np.save("score_names", X_score_name)
np.save("Y", Y_composer)
print(len(Y_composer))
print('encoding time {}s'.format(time() - ts))

print("Decoding dataset...")
X_score = np.load("X.npy")
X_score_name = np.load("score_names.npy")
Y_composer = np.load("Y.npy")
total = len(X_score)
ts = time()
for i, score in enumerate(X_score):
    if i % 100 == 0:
        print(i, '/', total, ':', score_name)
    score_name = X_score_name[i]
    composer_id = Y_composer[i]
    decode_score(score, "decode_test")
    print("decoded", COMPOSERS[composer_id], score_name)
    break
print("Done.")