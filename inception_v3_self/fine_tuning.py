import tensorflow as tf
import os
import numpy as np
import random

LABELS = {
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
MY_MODEL = './model.ckpt'

LEARNING_RATE = 0.01
BATCH = 100
STEP = 4000
EPOCH = 100

def get_random_bottleneck(path, file_num):
    bottleneck_values = []
    bottleneck_label = []
    for i in range(BATCH):
        label = random.randint(0, NUM_CLASS-1)
        category = LABELS[label]
        category_fullpath = os.path.join(path, category)
        index = random.randint(0, file_num - 1)
        file_name = os.path.basename(path) + '_' + category + '_' + str(index) + '.txt'
        bottleneck_path = os.path.join(category_fullpath, file_name)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        value = [float(x) for x in bottleneck_string.split(',')]
        label_one_hot = np.zeros(NUM_CLASS, dtype=np.float32)
        label_one_hot[label] = 1.0
        bottleneck_values.append(value)
        bottleneck_label.append(label_one_hot)
    return bottleneck_values, bottleneck_label

def get_test_bottleneck():
    bottleneck_values = []
    bottleneck_label = []

    for label in range(5):
        category = LABELS[label]
        category_path = os.path.join(TESTING_DATA, category)
        for files in os.listdir(category_path):
            files_path = os.path.join(category_path, files)
            with open(files_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            value = [float(x) for x in bottleneck_string.split(',')]
            label_one_hot = np.zeros(NUM_CLASS, dtype=np.float32)
            label_one_hot[label] = 1.0
            bottleneck_values.append(value)
            bottleneck_label.append(label_one_hot)
    return bottleneck_values, bottleneck_label

def main():
    # Define the Input
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    label_input = tf.placeholder(tf.float32, [None, NUM_CLASS], name='LableInput')

    # Define a full connect
    with tf.name_scope('training_full_net'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, NUM_CLASS], stddev=0.001))
        biases = tf.Variable(tf.zeros([NUM_CLASS]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        y = tf.nn.softmax(logits)

    # Define the loss
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= label_input)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    # Define the training step
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entroy_mean)

    # Calculate the accuracy
    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label_input, 1))
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEP):
            train_bottleneck_value, train_bottleneck_label = get_random_bottleneck(TRAINING_DATA, 360)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottleneck_value, label_input: train_bottleneck_label})

            if i % 50 == 0 or i + 1 == STEP:
                saver.save(sess, MY_MODEL, global_step=i)
                saver.export_meta_graph("./model.ckpt.meda.json", as_text=True)
                validation_bottleneck_value, validation_bottleneck_label = get_random_bottleneck(VALIDATING_DATA, 120)
                validation_accuracy = sess.run(evalution_step, feed_dict={bottleneck_input:validation_bottleneck_value, label_input:validation_bottleneck_label})
                print ("Step %d: validation accuracy = %.1f%%" % (i, validation_accuracy * 100))

        test_bottleneck_value, test_bottleneck_label = get_test_bottleneck()
        test_accuracy = sess.run(evalution_step, feed_dict={bottleneck_input: test_bottleneck_value, label_input: test_bottleneck_label})
        print ("Test accuracy = %.1f%%" % (test_accuracy * 100))

if __name__ == '__main__':
    main()