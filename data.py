import tensorflow as tf
import numpy as np

class Data:
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed = 0):
        self.seed = seed
        self.full_dataset_size = Data._dataset_size(filename_list)
        self.test_dataset_size = self.full_dataset_size - training_size
        
        self.filenames = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        self.num_batches = training_size // batch_size
        
        self.full_dataset = tf.data.TFRecordDataset(self.filenames).map(self._parse)
        self.full_dataset = self.full_dataset.shuffle(self.full_dataset_size, seed = seed, 
            reshuffle_each_iteration = True)
        
        self.training_dataset = self.full_dataset.take(training_size)
        training_iterator = self.training_dataset.batch(self.training_size).make_one_shot_iterator()
        
        self.test_dataset = self.full_dataset.skip(training_size).take(-1).batch(
            self.test_dataset_size)
        test_iterator = self.test_dataset.make_one_shot_iterator()
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.training_dataset = sess.run(training_iterator.get_next())
        self.test_dataset = sess.run(test_iterator.get_next())
        
        self.batch_num = 0
        
    def get_training_dataset_as_dict(self):
        return self.training_dataset

        
    def get_training_batch_as_dict(self):   
        batch_start_index = self.batch_num * self.batch_size
        batch_end_index = batch_start_index + self.batch_size
            
        batch = {'x': self.training_dataset['x'][batch_start_index: batch_end_index],
                 'y': self.training_dataset['y'][batch_start_index: batch_end_index],
                 'z': self.training_dataset['z'][batch_start_index: batch_end_index]}
        
        self.batch_num = (self.batch_num + 1 % self.num_batches)
        if(self.batch_num == 0):
            self.shuffle_training_dataset()
        
        return batch
        
    def get_test_dataset_as_dict(self):
        return self.test_dataset

        

    @staticmethod
    def _dataset_size(filename_list: list):
        count = 0
        for filename in filename_list:
            for _ in tf.python_io.tf_record_iterator(filename):
                count = count + 1
        
        return count
    
    @staticmethod
    def _parse(dataset: tf.data.TFRecordDataset):
        features = {'x': tf.FixedLenFeature(shape = (8,), dtype = tf.int64),
                    'y': tf.FixedLenFeature(shape = (8,), dtype = tf.int64),
                    'z': tf.FixedLenFeature(shape = (8,), dtype = tf.int64)}
        
        parsed_features = tf.parse_single_example(dataset, features)
        return parsed_features
    
    def shuffle_training_dataset(self):
        for key in self.training_dataset.keys():
            np.random.seed(seed = 0)
            np.random.shuffle(self.training_dataset[key])
    
"""
Code sandbox for figuring stuff out
"""
    
if __name__ == '__main__':
    
    dataset= Data(['test/test_data.tfrecords'], 10, 2, 0)
    print(dataset.get_training_dataset_as_tensor_dict()['x'][0:5])
    