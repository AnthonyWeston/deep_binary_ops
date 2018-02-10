import numpy as np
import unittest
from data_generator.data_generator import add_op
from generate_training_data import *


class TestGenerateTrainingDataHelperFunctions(unittest.TestCase):
    
    def test_data_row_returns_a_comma_separated_binary_string_with_x_y_and_result_z(self):
        subject_x = np.uint8(5) # 00000101
        subject_y = np.uint8(10) # 00001010
        subject_op = add_op
        
        expected_string = '0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1\n' # z = 00001111
        self.assertEquals(expected_string, data_row(subject_x, subject_y, subject_op))
        
    def test_csv_header_returns_a_csv_header_row(self):
        expected_string = 'x7,x6,x5,x4,x3,x2,x1,x0,y7,y6,y5,y4,y3,y2,y1,y0,z7,z6,z5,z4,z3,z2,z1,z0\n'
        self.assertEquals(expected_string, csv_header())