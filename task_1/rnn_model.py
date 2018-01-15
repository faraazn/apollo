from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Load training and eval data
X = np.load("X_rnn.npy").astype(np.float32)
Y = np.load("Y_rnn.npy").astype(np.int32)
train_set = np.load("train_0.p")
valid_set = np.load("valid_0.p")
test_set = np.load("test_0.p")
train_data = X[len(valid_set):len(valid_set)+len(train_set)]
train_labels = Y[len(valid_set):len(valid_set)+len(train_set)]
eval_data = X[:len(valid_set)]
eval_labels = Y[:len(valid_set)]
print(train_data.shape)
print(train_labels.shape)
print(eval_data.shape)
print(eval_labels.shape)
print(train_labels)

# Training Parameters
learning_rate = 0.001
training_steps = 1000
batch_size = 128
display_step = 200

# Network Parameters
NOTE_RANGE = 88
STEPS_PER_CUT = 48*4

num_input = NOTE_RANGE # MNIST data input (img shape: 28*28)
timesteps = STEPS_PER_CUT # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


# Define weights
weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def RNN(x, weights, biases):
	
	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
	
	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, timesteps, 1)
	
	# Define a lstm cell with tensorflow
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	
	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	
	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
	
	# Run the initializer
	sess.run(init)
	index_in_epoch = 0
	num_examples = len(train_data)
	print(train_data.shape)
	epochs_completed = 0
	
	for step in range(1, training_steps+1):
		start = index_in_epoch
		index_in_epoch += batch_size
		if index_in_epoch > num_examples:
			epochs_completed += 1
			print("epoch", epochs_completed)
			perm = np.arange(num_examples)
			np.random.shuffle(perm)
			train_data = train_data[perm]
			train_labels = train_labels[perm]
			start = 0
			index_in_epoch = batch_size
			assert batch_size <= num_examples
		end = index_in_epoch
		
		batch_x = train_data[start:end]
		batch_y = train_labels[start:end]
		
		# Reshape data to get STEPS_PER_CUT seq of NOTE_RANGE elements
		batch_x = batch_x.reshape((batch_size, timesteps, num_input))
		batch_y = np.eye(2)[batch_y]
		batch_y = batch_y.reshape((-1, 2))
		
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
			print("Step " + str(step) + ", Minibatch Loss= " + \
				"{:.4f}".format(loss) + ", Training Accuracy= " + \
				"{:.3f}".format(acc))
	
	print("Optimization Finished!")
	
	# Calculate accuracy for 5 midi test files
	test_len = 5
	test_data = eval_data.reshape((-1, timesteps, num_input))
	test_label = np.eye(2)[eval_labels[step % len(eval_labels)]]
	test_label = test_label.reshape((-1, 2))
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))