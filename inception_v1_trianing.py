from inception_v1 import *
import tensorflow as tf
import numpy as np
import cv2
import os

slim = tf.contrib.slim

checkpoint_file = "inception_v1.ckpt"
data_dir = "selected_data"
LOGDIR = "training_log"

BATCH_SIZE = 1 << 4
NUM_CLASSES = 10
LEARNING_RATE = 1e-4

TRAIN_STEPS = 12000

sampling_index = np.random.randint(0, NUM_CLASSES, (TRAIN_STEPS, BATCH_SIZE))


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


def sub_sampling(training_set, label_set, _index):
    sample_f = []
    sample_l = []

    for x in range(BATCH_SIZE):
    	temp_rand = np.random.randint(0, 20)
        sample_f.append(training_set[_index[x]][temp_rand])
        sample_l.append(label_set[_index[x]][temp_rand])

    return (sample_f, sample_l)


def get_train(training_set, label_set):
    while True:
        for _index in sampling_index:
            yield sub_sampling(training_set, label_set, _index)


# def constractive_loss()

del_files(data_dir)
training_set, label_set = extract_features(load_data(os.path.join(data_dir, "train")))

tf.reset_default_graph()
sess = tf.Session()

with tf.name_scope('input'):
    input_layer = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1024), name="features")
    labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES), name="labels")

y = slim.fully_connected(input_layer, 256, activation_fn=tf.nn.sigmoid, scope='fc/fc_0')
y = slim.fully_connected(y, NUM_CLASSES, activation_fn=tf.nn.sigmoid, scope='fc/fc_1')

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.name_scope("evaluation"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)   

train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))
train_writer.add_graph(sess.graph)
summary_op = tf.summary.merge_all()

if __name__ == '__main__':
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	get_train = get_train(training_set, label_set)

	for step in range(TRAIN_STEPS):
		(batch_xs, batch_ys) = get_train.next()
	
		summary_result, _ = sess.run([summary_op, train], feed_dict={input_layer: batch_xs, labels: batch_ys})
	
		if (step % 100 == 0) and (step * 200 / BATCH_SIZE > 10):
			_accuracy = sess.run(accuracy, feed_dict={input_layer: batch_xs, labels: batch_ys})
			if _accuracy > (0.999):
				save_path = saver.save(sess, "./model/model.ckpt")
				print("Model saved in file: %s" % save_path)

		train_writer.add_summary(summary_result, step)
		train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % step)
	
	train_writer.close()












