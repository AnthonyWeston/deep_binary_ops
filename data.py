import tensorflow as tf

class Data:
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed = 0):
        self.full_dataset_size = Data._dataset_size(filename_list)
        
        self.filenames = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        
        self.full_dataset = tf.data.TFRecordDataset(self.filenames).map(self._parse)
        self.full_dataset = self.full_dataset.shuffle(4096, seed = seed, reshuffle_each_iteration = True)
        
        self.training_dataset = self.full_dataset.take(training_size).batch(self.batch_size)
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        
        self.test_dataset = self.full_dataset.skip(training_size).take(-1)
        self.test_iterator = self.test_dataset.make_one_shot_iterator()
        

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
    
    
    
    
if __name__ == '__main__':
    
    dataset= Data(['test/test_data.tfrecords'], 10, 2, 0)
    print(dataset.full_dataset)
    print(dataset.training_dataset)
    print(type(dataset.test_dataset))
    
    with tf.Session() as sess:
        print(sess.run(dataset.test_iterator.get_next()))
    