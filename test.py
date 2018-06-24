from inception_v1 import *
import tensorflow as tf
import numpy as np
import cv2
import os

slim = tf.contrib.slim

checkpoint_file = "inception_v1.ckpt"
applied_ckpt = "./model/model.ckpt"
data_dir = "selected_data"

NUM_CLASSES = 10
BATCH_SIZE = 1


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def load_data(data_path):
	image_set = []
	for i in range(NUM_CLASSES):
		image_set.append([])

	for human_name in os.listdir(data_path):
		for frame_name in os.listdir(os.path.join(data_path, human_name)):
			image_set[int(human_name[0])].append(np.asarray(cv2.imread(os.path.join(data_path, human_name, frame_name))))

	return image_set


def onehot_encoding(_index, num_classes=NUM_CLASSES):
    label = np.zeros(shape=[num_classes], dtype='float32')
    label[int(_index) - 1] = 1
    return label


def extract_features(image_set):
	feat_set = []
	label_set = []
	for i in range(NUM_CLASSES):
		feat_set.append([])
		label_set.append([])

	height, width, channels = 224, 224, 3
	x = tf.placeholder(tf.float32, shape=(1, height, width, channels))
	arg_scope = inception_v1_arg_scope()
	with slim.arg_scope(arg_scope):
		logits, end_points = inception_v1(x, is_training=False, num_classes=1001)
		features = end_points['AvgPool_0a_7x7']

	sess  = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, checkpoint_file)

	for i in range(NUM_CLASSES):
		for j in range(20):
			image = np.expand_dims(image_set[i][j], 0)
			feat = np.squeeze(sess.run(features, feed_dict={x:image}))
			feat_set[i].append(feat)
			label_set[i].append(onehot_encoding(i))

	return (feat_set, label_set)


# def constractive_loss()

del_files(data_dir)
training_set, label_set = extract_features(load_data(os.path.join(data_dir, "test")))

tf.reset_default_graph()
sess = tf.Session()

with tf.name_scope('input'):
    input_layer = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1024), name="features")
    labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES), name="labels")

y = slim.fully_connected(input_layer, 256, activation_fn=tf.nn.sigmoid, scope='fc/fc_0')
y = slim.fully_connected(y, NUM_CLASSES, activation_fn=tf.nn.sigmoid, scope='fc/fc_1')


if __name__ == '__main__':
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, applied_ckpt)
	correct_count = 0

	for i in range(NUM_CLASSES):
		for j in range(20):
			prediction = sess.run(y, feed_dict={input_layer: np.expand_dims(training_set[i][j], 0)})
			if np.equal(np.argmax(prediction), np.argmax(label_set[i][j])):
				correct_count = correct_count + 1

	print("Accuracy:")
	print(1.0 * correct_count / 2)










