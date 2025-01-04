"""
this python file sums the function that transform 3-slot rgb arrays into 1 int, and vice versa
"""

import math, numpy as np

def rgb_to_256base(rgb_array): # the input is a 3-slot array with ints in the range [0,255], and the output is in the range [0, 1]
    return (rgb_array[0]*(256**2) + rgb_array[1]*(256) + rgb_array[2])/(256**3 -1)

def base256_to_rgb(value): # the input is int in the range [0, 1], the output is a 3-slot array with ints in the range [0,255]. if the number is greater than 1 - raise an error?
    if (value>1) or (value<0):
        print("IMPORTANT ERROR! value is not in the range [0,1]")
    integer = math.floor(value * (256**3 -1))
    b = integer%256
    integer = (integer-b)/256
    g = integer%256
    r = (integer-g)/256
    return np.array([r,g,b]).astype(np.uint8)
    # return np.array([r,g,b]).astype(np.uint8)

def concat_rgb_sum(rgb_array): # the input is 1d array with length 3. the output is 1 scalar. this is a general sum that's used
    return sum(rgb_array)/(255*3)

def base_sum_to_rgb(value): # here we use the "traditional" sum instead of the base 256.the input is int on the range [0,1] and the output is 3-slot array in the range [0,255]
    output = [0,0,0]
    integer_val = math.floor(value * 255 * 3)
    if integer_val <=255:
        return np.array([0, 0, integer_val]).astype(np.uint8)
    if integer_val <=255*2:
        return np.array([0, integer_val-255, 255]).astype(np.uint8)
    return np.array([integer_val, 255, 255]).astype(np.uint8)

def rgb_to_base_sum(value): #the input is 3-slot array with values between [0, 255] and the output is 1 number between [0,1]
    return sum(value)/(255*3)
