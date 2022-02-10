"""
    image_convertor.py: list of functions related to file conversion.
    namely, ascii to/from binary conversion functions for a given pgm file.
    
    References:
    1. https://en.wikipedia.org/wiki/Netpbm
    2. https://www.programiz.com/python-programming/methods/built-in/bytearray
    3. https://stackoverflow.com/questions/4218760/convert-binary-files-into-ascii-in-python 
    4. https://stackoverflow.com/questions/40082165/python-create-pgm-file
    5. https://learning-python.com/strings30.html   

@author: mohan_chimata    
"""

# built-in package imports
import os
import numpy as np

# converts the ascii pgm file into binary format.
def convert_ascii_to_bin(filename):
    bin_file = filename.replace('.pgm', '_b.pgm')

    with open(bin_file, 'wb') as img_bin:
        # tokenizing file data into a list of integers
        data = read_ascii_pgm_file(filename)

        width, height, max_pixel = data[0:3]
        
        # writing file header content to the file.
        pgm_header = f'P5{os.linesep}{width} {height}{os.linesep}{max_pixel}{os.linesep}'
        img_bin.write(bytearray(pgm_header, encoding='utf-8'))

        # converting the pixel data to numpy 2d array.
        pixel_data = np.array(data[3:])
        pixel_data = np.reshape(pixel_data, (height, width))
        
        # saving the pixel data row-by-row in binary encoded format. 
        for i in range(height):
            # extracts a data row of elements of size: width            
            pixel_row_data = list(pixel_data[i, :])
            img_bin.write(bytearray(pixel_row_data)) # saving pixels row data

    return bin_file

# converts the binary pgm file into ascii format.
def convert_bin_to_ascii(filename):
    ascii_filename = filename.replace('_b.pgm', '_copy.pgm')
    
    # reading bin_image file contents into list of lines.
    bin_img_lines = read_bin_pgm_file(filename)

    with open(ascii_filename, 'w') as img_ascii:
        # writing header(magic-number)
        img_ascii.write(f'P2{os.linesep}')
        
        # writing dimensions(width, height) and gray scale lines.
        for line in bin_img_lines[1:3]:
            img_ascii.write(str(line).strip("b'\\n") + os.linesep)
        
        # writing pixels data 
        for line in bin_img_lines[3:]:
            pixels_line = [str(line[i]) for i in range(len(line))]
            img_ascii.write(' '.join(pixels_line) + os.linesep)

    return ascii_filename

## ===== helper functions ====== ##

# reads ascii pgm file contents
# asserts file-type by checking magic-number and
# @returns the contents as list of integers except header magic-number.
def read_ascii_pgm_file(filename):
    with open(filename) as img_ascii:
        # tokenizing the file content into list of lines and ignoring comment lines
        line_wise_data = [line.strip().split() for line in img_ascii if not line.startswith('#')] 
        
        # first-line first-element should be the magic number, for instance P2 here.
        header = line_wise_data[0].pop(0) # removing the magic-number(P2 here).
        assert header == 'P2' # asserting file type.

        # converting data list of lists into a flat list of integers
        data = [int(element) for line in line_wise_data for element in line if element] 

    return data

# reads the binary pgm file contents
# asserts file-type by checking magic-number and
# returns the binary pgm file content that has been read.
def read_bin_pgm_file(filename):

    # reading bin_image file contents into list of lines.
    data = []
    with open(filename, 'rb') as img_bin:
        # reading entire file content into a list.
        data = img_bin.readlines()
    
    # asserting file type.
    assert 'P5' in str(data[0])

    return data
