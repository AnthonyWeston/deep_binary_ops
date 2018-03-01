import unittest
import tensorflow as tf
from data import Data
import numpy as np

class TestData(unittest.TestCase):
    
    sess = None
    test_filename_list = ['test/test_data.tfrecords']
    batch_size = 2
    
    @classmethod
    def setUpClass(cls):
        sess = tf.Session()
    
    def setUp(self):
        self.tfrecord_dataset = tf.data.TFRecordDataset(['test/test_data.tfrecords'])
        self.dataset= Data(TestData.test_filename_list, TestData.batch_size)
        
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
        self.assertEquals(expected_filename_list, subject.filenames)
            
    def test_the_data_model_is_created_with_a_batch_size(self):
        subject = self.dataset
        
        expected_batch_size = TestData.batch_size
        self.assertEquals(expected_batch_size, subject.batch_size)
        
        
        
        
        
        
        
        