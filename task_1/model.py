from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


NOTE_RANGE = 88
STEPS_PER_CUT = 48*4

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, STEPS_PER_CUT, NOTE_RANGE, 1])
	
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[4, 1], # 4 16th notes
		padding="same",
		activation=tf.nn.relu)
	
	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	# input size reduced to (96, 44, 32)
	
	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=32,
		kernel_size=[3, 3], # notes nearby and measure
		padding="same",
		activation=tf.nn.relu)
	
	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	# input size reduced to (48, 22, 32)
	
	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=64,
		kernel_size=[4, 4], # octave and measures
		padding="same",
		activation=tf.nn.relu)
	
	# Pooling Layer #3
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
	# input size reduced to (24, 11, 64)
	
	# Dense Layer
	pool3_flat = tf.reshape(pool3, [-1, 24*11*64]) # shape now (16896)
	dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu) # shape now (1024)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN) # shape now (1024)
	
	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=2) # shape now (2)
	
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	print("classes", predictions["classes"])
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		print("predict")
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	# Calculate Loss (for both TRAIN and EVAL modes)
	# onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
# 	print("onehot1", onehot_labels)
	onehot_labels = tf.one_hot(indices=labels, depth=2)
	print("onehot2", onehot_labels)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)
	
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	
	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	# Load training and eval data
	X = np.load("X_0.npy").astype(np.float32)
	Y = np.load("Y_0.npy").astype(np.int32)
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
	
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="./cnn_model")
	
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=100)
	
	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=2000,
		hooks=[logging_hook])
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()