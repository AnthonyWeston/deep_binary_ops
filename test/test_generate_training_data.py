import numpy as np
import unittest
from data_generator.data_generator import add_op
from generate_training_data import *


class TestGenerateTrainingDataHelperFunctions(unittest.TestCase):
    
    def test_generate_training_example_generates_an_example_with_the_x_values(self):
        subject = generate_training_example(101, 105)
        
        expected_value = dg.binary_array(101)
        np.testing.assert_array_equal(expected_value, subject.features.feature['x'].int64_list.value)