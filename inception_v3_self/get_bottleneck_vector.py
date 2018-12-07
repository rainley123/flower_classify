import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

# Name of bottleneck
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# Input image tensor
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# Directory of module
MODEL_DIR = './inception_dec_2015'

# The file of model
MODEL_FILE = 'tensorflow_inception_graph.pb'

# Directory of cache to save the feature vector
CACHE_DIR = './bottleneck'

# Date of image, I have saved the image as tfrecord
INPUT_DATA = './flower_photo'

# Get the feature vector of an image by inception_v3 model
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # Calculate the bottleneck tensor through the image_data
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # Make the shape from 4 to 1
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# Find/Build a feature vector(use run_bottleneck_image func)
def get_or_create_bottleneck(sess, image_data, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # Name of the bottleneck file
    bottleneck_name = category + '_' + label_name + '_' + str(index) + '.txt'
    bottleneck_dir = os.path.join(CACHE_DIR, category, label_name)

    # If the dir is not exist, just make it
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)
    bottleneck_path = os.path.join(bottleneck_dir, bottleneck_name)

    # If the bottleneck is not exist, just create and save it
    if not os.path.exists(bottleneck_path):
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

def main():
    with tf.Session() as sess:
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, name='', return_elements=
                                      [BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        for category_dir in os.listdir(INPUT_DATA):
            category_fullpath = os.path.join(INPUT_DATA, category_dir)
            for classfy_dir in os.listdir(category_fullpath):
                classfy_fullpath = os.path.join(category_fullpath, classfy_dir)
                index = 0
                for files in os.listdir(classfy_fullpath):
                    files_fullpath = os.path.join(classfy_fullpath, files)
                    print files_fullpath
                    image_data = gfile.FastGFile(files_fullpath, 'rb').read()
                    get_or_create_bottleneck(sess, image_data, classfy_dir, index, category_dir, jpeg_data_tensor, bottleneck_tensor)
                    index = index + 1

if __name__ == '__main__':
    main()


