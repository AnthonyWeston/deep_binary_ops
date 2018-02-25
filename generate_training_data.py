import data_generator.data_generator as dg
import os
import tensorflow as tf
from data_generator import *


UINT8_MAX = 255
dirname, _ = os.path.split(os.path.abspath(__file__))
OUTPUT_FOLDER = '{0}/training_data'.format(dirname)


def generate_training_data():
    
    for op in [dg.add_op, dg.multiply_op, dg.xor_op]:         
        for x in range(UINT8_MAX + 1):
            for y in range(UINT8_MAX + 1):
                generate_training_example(x, y, op)

def generate_training_example(x, y, op):
    x_values = dg.binary_array(x)
    y_values = dg.binary_array(y)
    z_values = dg.binary_array(op(x, y))
    
    training_example = tf.train.Example(features = tf.train.Features(feature = {
        'x': tf.train.Feature(int64_list = tf.train.Int64List(value = x_values)),
        'y': tf.train.Feature(int64_list = tf.train.Int64List(value = y_values)),
        'z': tf.train.Feature(int64_list = tf.train.Int64List(value = z_values))
        }))
    
    return training_example



if __name__ == '__main__':
    generate_training_data()