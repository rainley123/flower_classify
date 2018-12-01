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
LEARNING_RATE = 0.0000001
BATCH_SIZE = 30
SHUFFLE_BUFFER = 10000
CLASSES = 5
NUM_EPOCH = 100

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
    # create a train_dataset
    train_input_files = [TRAIN_TFRECORD]
    train_dataset = tf.data.TFRecordDataset(train_input_files)

    train_dataset = train_dataset.map(parser)
    train_dataset = train_dataset.map(resize_image)

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat(NUM_EPOCH)

    # define a iterator
    train_iterator = train_dataset.make_one_shot_iterator()
    image_batch, label_bacth = train_iterator.get_next()

    # create a test_dataset
    test_input_files = [TEST_TFRECORD]
    test_dataset =tf.data.TFRecordDataset(test_input_files)

    test_dataset = test_dataset.map(parser)
    test_dataset = test_dataset.map(resize_image)

    test_dataset = test_dataset.batch(BATCH_SIZE)

    test_iterator = test_dataset.make_one_shot_iterator()
    test_image_batch, test_label_batch = test_iterator.get_next()

    # create a validation_dataset
    val_input_files = [VAL_TFRECORD]
    val_dataset = tf.data.TFRecordDataset(val_input_files)

    val_dataset = val_dataset.map(parser)
    val_dataset = val_dataset.map(resize_image)

    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    # define the inputs of inception_v3
    input_images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_image')
    input_labels =tf.placeholder(tf.int64, [None], name='input_label')

    # define the model of inception_v3
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            input_images, num_classes=CLASSES
        )
    trainable_variables = get_trainable_variables()

    tf.losses.softmax_cross_entropy(tf.one_hot(input_labels, CLASSES), logits, weights=1.0)

    # define the step of training
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # tf.losses.softmax_cross_entropy(tf.one_hot(input_labels, CLASSES), logits)
    # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    # train_step = slim.learning.create_train_op(tf.losses.get_total_loss(), optimizer)

    # calculate the accuracy
    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), input_labels)
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
        step = 0
        while True:
            try:
                if step % 10 == 0:
                    saver.save(sess, TRAIN_FILE, global_step=step)

                    val_iterator = val_dataset.make_one_shot_iterator()
                    val_image_batch, val_label_batch = val_iterator.get_next()
                    val_accuracy = []
                    while True:
                        try:
                            evalution = sess.run(evalution_step, feed_dict=
                            {input_images: val_image_batch.eval(), input_labels: val_label_batch.eval()})
                            print 'Batch accuracy : %.1f%%' % float(evalution * 100.0)
                            val_accuracy.append(float(evalution))
                        except tf.errors.OutOfRangeError:
                            break
                    accuracy = tf.reduce_mean(val_accuracy)
                    print 'Step %d: accuracy = %.1f%%' % (step, accuracy.eval() * 100.0)

                sess.run(train_step, feed_dict={input_images: image_batch.eval(), input_labels: label_bacth.eval()})
                step = step + 1
            except tf.errors.OutOfRangeError:
                    break

    test_accuracy = []
    while True:
        try:
            result_evalution = sess.run(evalution_step, feed_dict= {input_images: test_image_batch.eval(), input_labels: test_label_batch.eval()})
            test_accuracy.append(result_evalution)
        except tf.errors.OutOfRangeError:
            break
    accuracy = tf.reduce_mean(test_accuracy)
    print 'The final accuracy of this model is %.1f%%' % accuracy.eval() * 100


if __name__ == '__main__':
    main()