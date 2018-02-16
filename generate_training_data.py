import data_generator.data_generator as dg
import os
import numpy as np


UINT8_MAX = 255
dirname, _ = os.path.split(os.path.abspath(__file__))
OUTPUT_FOLDER = '{0}/training_data'.format(dirname)


def generate_training_data():
    
    for op in [dg.add_op, dg.multiply_op, dg.xor_op]:
        continue



if __name__ == '__main__':
    generate_training_data()