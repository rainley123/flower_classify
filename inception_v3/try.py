import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

"""
input_data1 = [1,2,3,4,5,6]
input_data2 = [10,20,30,40]

dataset1 = tf.data.Dataset.from_tensor_slices(input_data1)
dataset1 = dataset1.shuffle(2).batch(2)
dataset1 = dataset1.repeat(5)
iterator1 = dataset1.make_one_shot_iterator()

dataset2 = tf.data.Dataset.from_tensor_slices(input_data2)
dataset2 = dataset2.batch(2)


x1 = iterator1.get_next()


input = tf.placeholder(tf.int64, [None])
y = input * 2
with tf.Session() as sess:
    step = 0
    while True:
        try:
            print sess.run(y,feed_dict= {input: x1.eval()})
            step = step + 1
            if step % 2 == 0:
                iterator2 = dataset2.make_one_shot_iterator()
                x2 = iterator2.get_next()
                while True:
                    try:
                        print sess.run(y, feed_dict={input: x2.eval()})
                    except tf.errors.OutOfRangeError:
                        break
        except tf.errors.OutOfRangeError:
            break
# with tf.Session() as sess:
#
#     accuracy = []
#     accuracy.append(0.01)
#     accuracy.append(0.02)
#     result = tf.reduce_mean(accuracy)
#     print sess.run(result)
"""

vgg = nets.vgg
logits, _ = vgg.vgg_16()
