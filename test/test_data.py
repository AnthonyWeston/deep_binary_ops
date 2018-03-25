import tensorflow as tf
from data import Data
import numpy as np
import os 

test_file_path = os.path.dirname(os.path.realpath(__file__))

class TestData(tf.test.TestCase):
    
    sess = None
    test_filename_list = [os.path.join(test_file_path, 'test_data.tfrecords')]
    batch_size = 2
    training_size = 10
    
    @classmethod
    def setUpClass(cls):
        TestData.sess = tf.Session()
    
    def setUp(self):
        self.tfrecord_dataset = tf.data.TFRecordDataset(['test/test_data.tfrecords'])
        self.dataset= Data(TestData.test_filename_list, training_size = TestData.training_size, 
            batch_size = TestData.batch_size, seed = 0)
        
    def test_the_data_model_parses_the_x_values_from_a_tfrecords_file(self):
        subject = self.tfrecord_dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['x'])
        
    def test_the_data_model_parses_the_y_values_from_a_tfrecords_file(self):
        subject = self.tfrecord_dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['y'])
        
    def test_the_data_model_parses_the_z_values_from_a_tfrecords_file(self):
        subject = self.tfrecord_dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['z'])
        
   
    def test_the_data_model_is_created_with_a_list_of_filenames(self):
        subject = self.dataset
        
        expected_filename_list = TestData.test_filename_list
        self.assertEqual(expected_filename_list, subject.filenames)
            
    def test_the_data_model_is_created_with_a_batch_size(self):
        subject = self.dataset
        
        expected_batch_size = TestData.batch_size
        self.assertEqual(expected_batch_size, subject.batch_size)
        
    def test_the_data_model_is_created_with_a_training_set_size(self):
        subject = self.dataset
        
        expected_batch_size = TestData.training_size
        self.assertEqual(expected_batch_size, subject.training_size)
        
    def test_the_data_model_is_split_into_a_training_dataset(self):
        subject = self.dataset.training_dataset
        
        expected_class = type(tf.data.TFRecordDataset(TestData.test_filename_list).take(-1))
        self.assertEqual(expected_class, type(subject))
        
    def test_the_data_model_is_split_into_a_test_dataset(self):
        subject = self.dataset.test_dataset
        
        expected_class = type(tf.data.TFRecordDataset(TestData.test_filename_list).take(-1).batch(2))
        self.assertEqual(expected_class, type(subject))
        
    def test_the_data_model_can_count_the_number_of_records_in_a_tfrecords_file(self):
        subject = TestData.test_filename_list
        
        expected_records = 16
        self.assertEqual(expected_records, Data._dataset_size(subject))
        
    def test_the_data_model_does_count_the_number_of_records_in_a_tfrecords_file(self):
        subject = self.dataset
        
        expected_records = 16
        self.assertEqual(expected_records, self.dataset.full_dataset_size)
        
    def test_the_data_model_gets_a_training_batch_as_a_dict_of_tensors(self):
        subject = self.dataset.get_training_batch_as_tensor_dict()['x']
        
        expected_class = tf.Tensor
        self.assertEqual(expected_class, type(subject))
        
    def test_the_data_model_gets_the_test_dataset_in_a_single_batch(self):
        dataset = self.dataset
        subject = dataset.get_test_dataset_as_tensor_dict()['x']
        
        expected_batch_size = dataset.test_dataset_size
        actual_batch_size = TestData.sess.run(subject)
        self.assertAllEqual(expected_batch_size, actual_batch_size.shape[0])
        
        
        
        
        
        
        
        