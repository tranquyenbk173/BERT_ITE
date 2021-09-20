import numpy as np
import tensorflow as tf
from src import settings
import subprocess
import os

if __name__ == "__main__":
    a = [1.0, 2.0, 0.0, 5]
    print(np.count_nonzero(a))
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    d = [[1, 2, 3],
         [2, 3, 4]]
    e = [[1, 1, 1],
         [0, 1, 3]]

    c = tf.divide(tf.reduce_sum(a), 4)

    with tf.Session() as sess:
        print(sess.run(tf.where(a == 2.0, ), {x: a, y: e}))
    print(list(sess.run((tf.shape(y)), {x: d, y: e})) + list(sess.run((tf.shape(x)), {x: d, y: e})[1:]))
    # dir_path = settings.DATA_ROOT_PATH + 'site_data/recobell/scene_1/'
    # # subprocess.call(['bash', 'bin/split.sh', dir_path])
    # for partition_name in sorted(os.listdir(dir_path + 'partitioned_train_data/')):
    #     print(partition_name)
