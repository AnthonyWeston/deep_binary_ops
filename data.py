import tensorflow as tf

class Data:
    
    @staticmethod
    def _parse(dataset: tf.data.TFRecordDataset):
        features = {'x': tf.FixedLenFeature(shape = (8,), dtype = tf.int64),
                    'y': tf.FixedLenFeature(shape = (8,), dtype = tf.int64),
                    'z': tf.FixedLenFeature(shape = (8,), dtype = tf.int64)}
        
        parsed_features = tf.parse_single_example(dataset, features)
        return parsed_features
    
    
    
    
if __name__ == '__main__':
    
    data = tf.data.TFRecordDataset(['test/test_data.tfrecords'])
    parsed_data = data.map(Data._parse)
    
    iterator = parsed_data.make_one_shot_iterator()

    with tf.Session() as sess:
        for i in range(4):
            value = sess.run(iterator.get_next())
            print(value)
    