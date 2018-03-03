import unittest
import os
from abstract_model_builder import AbstractModelBuilder

test_file_path = os.path.dirname(os.path.realpath(__file__))


class TestAbstractModelBuilder(unittest.TestCase):
    
    test_filename_list = [os.path.join(test_file_path, 'test_data.tfrecords')]
    training_size = 10
    
    def setUp(self):
        self.modelbuilder = AbstractModelBuilder(filename_list = TestAbstractModelBuilder.test_filename_list,
            training_size = TestAbstractModelBuilder.training_size)
    
    def test_the_abstract_model_builder_is_created_with_a_list_of_data_filenames(self):
        subject = self.modelbuilder
        
        expected_filename_list = TestAbstractModelBuilder.test_filename_list
        self.assertEqual(expected_filename_list, subject.filename_list) 
        
    def test_the_abstract_model_builder_is_created_with_a_training_set_size(self):
        subject = self.modelbuilder
        
        expected_training_size = TestAbstractModelBuilder.training_size
        self.assertEqual(expected_training_size, subject.training_size)