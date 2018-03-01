import unittest
import tensorflow as tf
from data import Data
import numpy as np

class TestData(unittest.TestCase):
    
    sess = None
    
    @classmethod
    def setUpClass(cls):
        sess = tf.Session()
    
    def setUp(self):
        self.dataset = tf.data.TFRecordDataset(['test/test_data.tfrecords'])
        
    def test_the_data_model_parses_the_x_values_from_a_tfrecords_file(self):
        subject = self.dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['x'])
        
    def test_the_data_model_parses_the_y_values_from_a_tfrecords_file(self):
        subject = self.dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['y'])
        
    def test_the_data_model_parses_the_z_values_from_a_tfrecords_file(self):
        subject = self.dataset.map(Data._parse)
        
        iterator = subject.make_one_shot_iterator()
        first_row_values = iterator.get_next()
        
        expected_array = np.ndarray([0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected_array, first_row_values['z'])