"""
    main.py: entry function which takes user cmd args and process 
    according to the chosen menu-option.
    
    References:
    1. http://www.cs.uakron.edu/~duan/class/635/projects/project2/project2.htm

@author: mohan_chimata
"""

# built-in package imports
import sys

# user imports
from image_convertor import *
from svd import *

def main(argv):
  try:
    # checks cmd args and raises error accordingly.
    validate_arguments(argv) 
    
    # basic cmd-line argument validations.
    selectedOption = int(argv[1])
    
    # menu and user options.
    if (selectedOption == 1): # ascii-pgm image to binary conversion
        bin_file = convert_ascii_to_bin(argv[2])
        print('created bin-file: ', bin_file)
        
    elif (selectedOption == 2): # bin-pgm image to ascii convertion
        ascii_file_name = convert_bin_to_ascii(argv[2])
        print('created ascii_copy file: ', ascii_file_name)

    elif (selectedOption == 3): # svd to approximated image. 
        if (len(argv) < 5):
          raise ValueError('Insufficient arguments. Usage: $ python main.py 3 header.txt svd.txt img_compressing_rank')   
        elif not argv[4].isdigit() or int(argv[4]) <= 0:
          raise ValueError('Invalid usage. rank should be an +ve non-zero integer')
        
        # unapcking header and svd txt files from args
        header_txt, svd_txt = argv[2:4]
        svd_to_compressed_image(header_txt, svd_txt, int(argv[4]))        

    elif (selectedOption == 4): # compressed image.pgm.SVD to readable image.
        convert_svd_compressed_bin_image_to_ascii(argv[2])

    elif (selectedOption == 5): # create header.txt and svd.txt files for pgm-file passed.        
        generate_header_svd_files_for_pgm_image(argv[2])

    elif (selectedOption == 6): # generated images for range of ranks using option 3 & 4.
        generate_images_by_rank_range(argv[2], argv[3:])

    else:
      print('Invalid option..!! please check menu below.')
      display_menu() # displays menu options.

  except Exception as e:
    print('Oops, exception: ', e)


def display_menu():
  print('''Menu:
  1. Convert pgm image from ASCII to binary.
  Usage: $ python main.py 1 ascii_pgm_file_with_relative_path
  2. Convert pgm image from binary to ASCII.
  Usage: $ python main.py 2 bin_pgm_file_with_relative_path
  3. Generate approxiated binary image for given rank using SVD.
  Usage: $ python main.py 3 header.txt svd.txt img_compressing_rank
  4. Generate pgm image from binary approximated image.
  Usage: $ python main.py 4 image_b.pgm.SVD
  5. Generate header.txt and svd.txt files for given pgm image.
  Usage: $ python main.py 5 image_pgm_file
  6. Generate images of ranks (all if none passed) by performing 3 and 4 options.
  Usage: $ python main.py 6 image_pgm_file (rank-p, rank-q, ...)''')

def validate_arguments(argv):
    error_msg = None
    if (len(argv) < 2):
      error_msg = 'No input file passed, exiting..'
    elif not argv[1].isdigit():
      error_msg = 'Invalid option. poisitve integer expected as menu option.'
    
    if error_msg:      
      display_menu() # displaying menu for usage details.
      raise ValueError(error_msg)

if __name__ == '__main__':
    main(sys.argv)
