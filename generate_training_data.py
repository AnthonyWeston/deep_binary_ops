import data_generator.data_generator as dg
import os
import tensorflow as tf
import random

UINT8_MAX = 255
dirname, _ = os.path.split(os.path.abspath(__file__))
OUTPUT_FOLDER = '{0}/training_data'.format(dirname)


def generate_training_data(limit: int = UINT8_MAX, operations = dg.OPERATIONS, 
    output_directory: str = 'training_data/'):


    for op in operations:
        training_examples = []

        output_filename = '{0}{1}_data.tfrecords'.format(output_directory, op.__name__)  
        with tf.python_io.TFRecordWriter(output_filename) as writer:       
            for x in range(limit + 1):
                for y in range(limit + 1):
                    training_example = generate_training_example(x, y, op)
                    training_example_str = training_example.SerializeToString()
                    training_examples.append(training_example_str)
        
        random.seed(0)
        random.shuffle(training_examples)
        training_examples = training_examples[:2**14]
        
        for example in training_examples:
            writer.write(example)
            
        print(len(training_examples))
        
        

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
    generate_training_data(UINT8_MAX, dg.OPERATIONS, 'test/')