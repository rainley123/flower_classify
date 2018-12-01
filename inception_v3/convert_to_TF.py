import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2

TRAIN_FLOWER = "./train_flower"
TEST_FLOWER = "./test_flower"
VAL_FLOWER = "./validation_flower"
TRAIN_TFRECORD_FILE = "./train_flower.tfrecords"
TEST_TFRECORD_FILE = "./test_flower.tfrecords"
VAL_TFRECORD_FILE = "./validation_flower.tfrecords"

LABELS = {
    'daisy':0,
    'dandelion':1,
    'rose':2,
    'sunflower':3,
    'tulip':4
}

def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def read_files(path):
    flower_dir = []
    flower_image = []
    flower_label = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            flower_dir.append(dir)

    for dir in flower_dir:
        for files in os.listdir(os.path.join(path, dir)):
            flower_image.append(os.path.join(dir, files))
            flower_label.append(LABELS[dir])

    return flower_dir, flower_image, flower_label

def get_info(path, flower_image):
    image_path = os.path.join(path, flower_image)
    image = cv2.imread(image_path)
    print image_path
    height, width, channel = image.shape[:3]
    image_raw_data = image.tobytes()
    return image_raw_data, width, height, channel

def convert_to_TF(image_raw_data, label, width, height, channel):
    example = tf.train.Example(features=tf.train.Features(feature={
        'width': _int64_feature(width),
        'height': _int64_feature(height),
        'channel': _int64_feature(channel),
        'image_raw_data': _bytes_feature(image_raw_data),
        'label': _int64_feature(label)
    }))
    return example

def main():
    writer1 = tf.python_io.TFRecordWriter(TRAIN_TFRECORD_FILE)
    writer2 = tf.python_io.TFRecordWriter(TEST_TFRECORD_FILE)
    writer3 = tf.python_io.TFRecordWriter(VAL_TFRECORD_FILE)

    flower_dir, flower_image, flower_label = read_files(TRAIN_FLOWER)
    for i, image in enumerate(flower_image):
        image_raw_data, width, height, channel = get_info(TRAIN_FLOWER, image)
        label = flower_label[i]
        print i
        example = convert_to_TF(image_raw_data, label, width, height, channel)
        writer1.write(example.SerializeToString())
    writer1.close()

    flower_dir, flower_image, flower_label = read_files(TEST_FLOWER)
    for i, image in enumerate(flower_image):
        image_raw_data, width, height, channel = get_info(TEST_FLOWER, image)
        label = flower_label[i]
        print i
        example = convert_to_TF(image_raw_data, label, width, height, channel)
        writer2.write(example.SerializeToString())
    writer2.close()

    flower_dir, flower_image, flower_label = read_files(VAL_FLOWER)
    for i, image in enumerate(flower_image):
        image_raw_data, width, height, channel = get_info(VAL_FLOWER, image)
        label = flower_label[i]
        print i
        example = convert_to_TF(image_raw_data, label, width, height, channel)
        writer3.write(example.SerializeToString())
    writer3.close()

if __name__ == '__main__':
    main()