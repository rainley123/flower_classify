# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

# load the inception_v3 model
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

TRAIN_TFRECORD = "./train_flower.tfrecords"
TEST_TFRECORD = "./test_flower.tfrecords"
VAL_TFRECORD = "./validation_flower.tfrecords"
TRAIN_FILE = './save_model'
CKPT_FILE = './inception_v3.ckpt'

# parameter of the training
LEARNING_RATE = 0.00001
BATCH_SIZE = 10
SHUFFLE_BUFFER = 10000
CLASSES = 5
EPOCH = 100

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINING_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'image_raw_data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    width = features['width']
    height = features['height']
    channel = features['channel']
    image_raw_data = features['image_raw_data']
    label = features['label']

    image = tf.decode_raw(image_raw_data, tf.uint8)
    image = tf.reshape(image, [height, width, channel])

    return image, label

def resize_image(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_resized = tf.image.resize_images(image, [299, 299], method=0)
    image_resized = tf.reshape(image_resized, [299, 299, 3])
    return image_resized, label

# get the parameter of Google model
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    # enum all the parameter of inception_v3
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINING_SCOPES.split(",")]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
        return variables_to_train

def main():
    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)
    dataset = dataset.map(resize_image)
    dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()

    # define the model of inception_v3
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            image_batch, num_classes=CLASSES
        )
    trainable_variables = get_trainable_variables()

    tf.losses.softmax_cross_entropy(tf.one_hot(label_batch, CLASSES), logits, weights=1.0)

    # define the step of training
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # calculate the accuracy
    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_batch)
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print 'Loading tuned variables from %s' % CKPT_FILE
        load_fn(sess)
        for i in range(EPOCH):
            sess.run(iterator.initializer, feed_dict={input_files: TRAIN_TFRECORD})
            while True:
                try:
                    sess.run(train_step)
                except tf.errors.OutOfRangeError:
                    break
            saver.save(sess, TRAIN_FILE, global_step=i)
            sess.run(iterator.initializer, feed_dict={input_files: VAL_TFRECORD})
            accuracy_list = []
            while True:
                try:
                    accuracy = sess.run(evalution_step)
                    accuracy_list.append(float(accuracy))
                except tf.errors.OutOfRangeError:
                    break
            val_accuracy = tf.reduce_mean(accuracy_list)
            print ("EPOCH %d : Validation accuracy = %.1f%%" % (i, val_accuracy.eval() * 100))

        sess.run(iterator.initializer, feed_dict={input_files: TEST_TFRECORD})
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