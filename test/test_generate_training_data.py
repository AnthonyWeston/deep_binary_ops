import numpy as np
import unittest
from data_generator.data_generator import add_op
from generate_training_data import *


class TestGenerateTrainingDataHelperFunctions(unittest.TestCase):
    
    def test_generate_training_example_generates_an_example_with_the_x_values(self):
        subject = generate_training_example(101, 105, add_op)
        
        expected_value = dg.binary_array(101)
        np.testing.assert_array_equal(expected_value, subject.features.feature['x'].int64_list.value)
        
    def test_generate_training_example_generates_an_example_with_the_y_values(self):
        subject = generate_training_example(101, 105, add_op)
        
        expected_value = dg.binary_array(105)
        np.testing.assert_array_equal(expected_value, subject.features.feature['y'].int64_list.value)
        
    def test_generate_training_example_generates_an_example_with_the_result_of_applying_the_operation_to_x_and_y(self):
        subject = generate_training_example(101, 105, add_op)
        operation = add_op
        
        expected_value = dg.binary_array(operation(101, 105))
        np.testing.assert_array_equal(expected_value, subject.features.feature['z'].int64_list.value)