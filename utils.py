import csv
import numpy as np
import tensorflow as tf

from keras.utils import to_categorical
from keras.preprocessing import sequence

# Yay! Helper functions!
def load_data(batch_size, length, is_training=True):
	if is_training:
		with open('data/train.csv', 'r') as f:
			reader = csv.reader(f)
			X_train = [list(map(int,i)) for i in reader]

		with open('data/train_labels.csv', 'r') as f:
			reader = csv.reader(f)
			y_train = [list(map(int,i)) for i in reader]
		y_train = np.expand_dims(y_train[0], 1)
		y_train = to_categorical(y_train)


		X_train = sequence.pad_sequences(X_train, maxlen=length)

		X_val = X_train[:len(X_train)//4]
		y_val = y_train[:len(y_train)//4]

		X_train = X_train[len(X_train)//4:]
		y_train = y_train[len(y_train)//4:]

		num_tr_batch = len(X_train) // batch_size
		num_val_batch = len(X_val) // batch_size

		return X_train, y_train, num_tr_batch, X_val, y_val, num_val_batch

	else:
		with open('data/test.csv', 'r') as f:
			reader = csv.reader(f)
			X_test = [list(map(int,i)) for i in reader]

		with open('data/test_labels.csv', 'r') as f:
			reader = csv.reader(f)
			y_test = [list(map(int,i)) for i in reader]
		y_test = np.expand_dims(y_test[0], 1)
		y_test = to_categorical(y_test)

		X_test = sequence.pad_sequences(X_test, maxlen=length)

		num_te_batch = len(X_test) // batch_size

		return X_test, y_test, num_te_batch

def get_batch_dataset(batch_size, length, num_threads):
	trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(batch_size, length, is_training=True)
	data_queues = tf.train.slice_input_producer([trX, trY])
	X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
								  batch_size=batch_size,
								  capacity=batch_size*64,
								  min_after_dequeue=batch_size*32,
								  allow_smaller_final_batch=False)
	return X,Y