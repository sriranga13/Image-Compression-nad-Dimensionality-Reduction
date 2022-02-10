'''
    utils.py
    Contains list of michellaneous helper functions.

    References:
    1. https://www.programiz.com/python-programming/methods/string/rfind
'''
# built-in package imports
import os

# constants
OUTPUT_DIR = 'test'

# helper functions

# gets the out directory file-path for given file.
def remove_file_extension(filename):
    file_name_without_extension = filename[:filename.rfind('.')] # subtring until file-extension    
    return file_name_without_extension

def get_header_and_svd_txt_filenames(filename):
    file_without_extension = remove_file_extension(filename)
    header_file = f'{file_without_extension}_header.txt'        
    svd_file = f'{file_without_extension}_SVD.txt'
    return header_file, svd_file
