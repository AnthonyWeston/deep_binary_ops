import numpy as np



def binary_array(number: np.uint8):
    return np.unpackbits(np.array(number, dtype = np.uint8))

def add_op(number1: np.uint8, number2: np.uint8):
    return np.add(number1, number2)

def multiply_op(number1: np.uint8, number2: np.uint8):
    return np.multiply(number1, number2)

def xor_op(number1: np.uint8, number2: np.uint8):
    return np.bitwise_xor(number1, number2)