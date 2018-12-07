import tensorflow as tf
import os
import numpy as np
import random

LABELS = {
    'daisy':0,
    'dandelion':1,
    'rose':2,
    'sunflower':3,
    'tulip':4
}

LABELS_RES = {
    0: 'daisy',
    1: 'dandelion',
    2: 'rose',
    3: 'sunflower',
    4: 'tulip'
}

# Node of bottleneck
BOTTLENECK_TENSOR_SIZE = 2048
NUM_CLASS = 5

# Path of data
TRAINING_DATA = './bottleneck/train_flower'
VALIDATING_DATA = './bottleneck/validation_flower'
TESTING_DATA = './bottleneck/test_flower'
BOTTLENECK_PATH = './bottleneck'
MY_MODEL = './save_model'

LEARNING_RATE = 0.01
BATCH = 100
SHUFFLE_BUFFER = 10000
EPOCH = 100

def create_tensor(bottleneck_path, classfy_dir):
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    label = LABELS[classfy_dir]
    bottleneck_values.append(label)

    bottleneck_values_and_label = bottleneck_values
    return bottleneck_values_and_label

def get_bottleneck_array(path):
    bottleneck_array = []

    for classfy_dir in os.listdir(path):
        classfy_fullpath = os.path.join(path, classfy_dir)
        for files in os.listdir(classfy_fullpath):
            bottleneck_path = os.path.join(classfy_fullpath, files)
            bottleneck_values_and_label = create_tensor(bottleneck_path, classfy_dir)
            bottleneck_array.append(bottleneck_values_and_label)
    return bottleneck_array

def get_bottleneck_label(bottleneck_values_and_label):
    bottleneck_values, bottleneck_label = tf.split(bottleneck_values_and_label, [BOTTLENECK_TENSOR_SIZE, 1])
    bottleneck_label = tf.cast(bottleneck_label, dtype=tf.int64)
    bottleneck_label = tf.one_hot(bottleneck_label, NUM_CLASS)
    bottleneck_label = tf.squeeze(bottleneck_label)
    return bottleneck_values, bottleneck_label

def main():
    """
    Prepare the  batch
    """
    train_bottleneck_array = get_bottleneck_array(TRAINING_DATA)
    val_bottleneck_array = get_bottleneck_array(VALIDATING_DATA)
    test_bottleneck_array = get_bottleneck_array(TESTING_DATA)

    Input_data = tf.placeholder(dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(Input_data)
    dataset = dataset.map(get_bottleneck_label)
    dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH)

    # Define a iterator
    iterator = dataset.make_initializable_iterator()
    bottleneck_values, bottleneck_label = iterator.get_next()

    # Define a full connect
    with tf.name_scope('training_full_net'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, NUM_CLASS], stddev=0.001))
        biases = tf.Variable(tf.zeros([NUM_CLASS]))
        logits = tf.matmul(bottleneck_values, weights) + biases
        y = tf.nn.softmax(logits)

    # Define the loss
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=bottleneck_label)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    # Define the training step
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entroy_mean)

    # Calculate the accuracy
    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(bottleneck_label, 1))
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(EPOCH):
            sess.run(iterator.initializer, feed_dict={Input_data: train_bottleneck_array})
            while True:
                try:
                    sess.run(train_step)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(iterator.initializer, feed_dict={Input_data: val_bottleneck_array})
            accuracy_list = []
            while True:
                try:
                    accuracy = sess.run(evalution_step)
                    accuracy_list.append(float(accuracy))
                except tf.errors.OutOfRangeError:
                    break
            val_accuracy = tf.reduce_mean(accuracy_list)
            print ("EPOCH %d : validation accuracy = %.1f%%" % (i, val_accuracy.eval() * 100))

        sess.run(iterator.initializer, feed_dict={Input_data: test_bottleneck_array})
        accuracy_list = []
        while True:
            try:
                accuracy = sess.run(evalution_step)
                accuracy_list.append(float(accuracy))
            except tf.errors.OutOfRangeError:
                break
        test_accuracy = tf.reduce_mean(accuracy_list)
        print ("Final test accuracy = %.1f%%" % (test_accuracy.eval() * 100))

if __name__ == '__main__':
    main()