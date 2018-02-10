import numpy as np
import unittest
from ..data_generator import *

class TestDataGenerator(unittest.TestCase):
    
    
    def test_the_data_generator_converts_a_numpy_byte_to_an_array_of_bits(self):
        subject = np.uint8(5)
        
        expected_array = np.array([0,0,0,0,0,1,0,1], dtype = np.uint8)
        np.testing.assert_array_equal(expected_array, convert_to_binary_array(subject))
        
