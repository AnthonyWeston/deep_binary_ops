import data_generator.data_generator as dg
import os
import numpy as np


UINT8_MAX = 255
dirname, filename = os.path.split(os.path.abspath(__file__))
OUTPUT_FOLDER = f'{dirname}/training_data'


def generate_training_data():
    
    for op in [dg.add_op, dg.multiply_op, dg.xor_op]:
        with open(f'{OUTPUT_FOLDER}/{op.__name__}.csv', 'w') as output_file:
            output_file.write(csv_header())
            
            for x in range(UINT8_MAX + 1):
                for y in range(UINT8_MAX + 1):
                    output_file.write(data_row(np.uint8(x), np.uint8(y), op))

def data_row(x: np.uint8, y: np.uint8, op):
    z = op(x, y)
    
    x_binary, y_binary, z_binary = (dg.binary_array(x), dg.binary_array(y), dg.binary_array(z))
                                                            
    x_str, y_str, z_str = (dg.comma_separated_string(x_binary), dg.comma_separated_string(y_binary), 
                           dg.comma_separated_string(z_binary))                                                                              

    data_row = ','.join([x_str, y_str, z_str]) + '\n'
    
    return data_row

def csv_header():
    header = ''
    for var_name in ['x', 'y', 'z']:
        for subscript in range(7, -1, -1):
            header += f'{var_name}{subscript}'
            header += ',' if subscript > 0 else ''
        
        header += ',' if var_name != 'z' else '\n'
        
    return header



if __name__ == '__main__':
    generate_training_data()