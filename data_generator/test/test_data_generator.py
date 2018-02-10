import numpy as np
import unittest
from ..data_generator import *


class TestDataGenerator(unittest.TestCase):
    
    
    def test_the_data_generator_converts_a_numpy_byte_to_an_array_of_bits(self):
        subject = np.uint8(5)
        
        expected_array = np.array([0,0,0,0,0,1,0,1], dtype = np.uint8)
        np.testing.assert_array_equal(expected_array, binary_array(subject))
        

    def test_the_data_generator_adds_two_numpy_bytes(self):
        subject1 = np.uint8(100)
        subject2 = np.uint8(201)
        
        expected_number = np.uint8(100 + 201)
        self.assertEquals(expected_number, add_op(subject1, subject2))
        
    def test_the_data_generator_multiplies_two_numpy_bytes(self):
        subject1 = np.uint8(100)
        subject2 = np.uint8(201)
        
        expected_number = np.uint8(100 * 201)
        self.assertEquals(expected_number, multiply_op(subject1, subject2))
        
    def test_the_data_generator_computes_bitwise_xor_of_two_numpy_bytes(self):
        subject1 = np.uint8(100)
        subject2 = np.uint8(201)
        
        expected_number = np.uint8(100 ^ 201)
        self.assertEquals(expected_number, xor_op(subject1, subject2))
        
    def test_the_data_generator_converts_a_numpy_array_to_a_comma_separated_string(self):
        subject = np.arange(5, dtype = np.uint8)
        
        expected_string = '0,1,2,3,4'
        self.assertEquals(expected_string, comma_separated_string(subject))