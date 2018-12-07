import os
from tensorflow.python import pywrap_tensorflow
# import tensorflow as tf
#
# MODEL = './save_model-3999.meta'
#
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(MODEL)
#     saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))
#     print (sess.run('LableInput:0'))

reader = pywrap_tensorflow.NewCheckpointReader('./model.ckpt-3999')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))