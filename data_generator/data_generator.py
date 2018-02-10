import numpy as np



def convert_to_binary_array(number: np.uint8):
    return np.unpackbits(np.array(number, dtype = np.uint8))

