from data import Data
import unittest
import os
from abstract_model import AbstractModel

test_file_path = os.path.dirname(os.path.realpath(__file__))


class TestAbstractModel(unittest.TestCase):
    
    test_filename_list = [os.path.join(test_file_path, 'test_data.tfrecords')]
    
    def setUp(self):
        self.model = AbstractModel(TestAbstractModel.test_filename_list)
    
    def test_the_abstract_model_is_created_with_a_list_of_data_filenames(self):
        subject = self.model
        
        expected_filename_list = TestAbstractModel.test_filename_list
        self.assertEqual(expected_filename_list, subject.filename_list) 
        
