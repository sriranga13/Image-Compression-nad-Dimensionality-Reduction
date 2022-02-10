"""
    svd.py: list of functions related svd based image-conversion operations. 
    
    References:
    1. https://web.stanford.edu/class/cs168/l/l9.pdf
    2. https://numpy.org/doc/stable/reference/routines.io.html#raw-binary-files
    3. https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html
    4. https://www.py4u.net/discuss/19530 (Answer#5)
    5. https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    6. https://stackoverflow.com/questions/47542927/python-code-svd-with-numpy    
    7. https://www.youtube.com/watch?v=xy3QyyhiuY4
    8. https://stackoverflow.com/questions/44691524/write-a-2d-array-to-a-csv-file-with-delimiter
    9. https://pythonbasics.org/matplotlib-line-chart/
    10. https://datascienceparichay.com/article/plot-a-line-chart-in-python-with-matplotlib/
    11. https://www.kite.com/python/answers/how-to-show-two-figures-at-once-in-matplotlib-in-python
    12. https://www.programiz.com/python-programming/writing-csv-files

@author: mohan_chimata
"""

# built-in package imports
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

# user imports
import image_convertor as img
import utils

# compresses the given pgm image to rank-k using SVD.
# and stores the matrix in binary format.
def svd_to_compressed_image(header_file, svd_file, k):
    
    # reads header.txt file contents
    width, height, max_pixel = read_header_file(header_file)
    
    # reads svd.txt file contents as 1D numpy.array
    svd_data = read_svd_file(svd_file)

    U, S, V_transpose = extract_svd_matrices(svd_data, height, width, k)
    
    # calculate appr error => sigma(k+1) element
    appr_error = calculate_approximation_error(height, width, S, k)

    # extracting compressed U, sigma and V.
    # i.e., copying U, S, V to new matrices, and resizing based on rank.           
    new_U = U[:, :k]
    new_S = S[:k, :k]
    new_V_transpose = V_transpose[:k, :]
    
    # generating A from resized U, S, V matrices using SVD formula.
    compressed_A = np.matmul(np.matmul(new_U, new_S), new_V_transpose)

    # print('\nresized A:')
    # for r in compressed_A:
    #     print(r)
    # print()

    # this is test data
    bin_file = save_compressed_pgm_file('image', (width, height, max_pixel, k), compressed_A)
    # convert_svd_compressed_bin_image_to_ascii(bin_file)
    return (bin_file, appr_error)

# convert the svd-compressed binary image to readable pgm image format.
def convert_svd_compressed_bin_image_to_ascii(bin_file):
    pgm_header, rank, image = read_svd_compressed_bin_image(bin_file)

    img_rank_file = f'image_{rank}.pgm'
    
    # writing the approximated pgm file back to file.
    with open(img_rank_file, 'w') as img_rank_ascii:
        img_rank_ascii.write(pgm_header)

        # saving the pixel data 
        pixels_row = []
        for row in image:
            # concatenating pixel row values with whitespace: row-strings.
            pixels_row.append(' '.join(map(str, row)))

        # appending row-strings by new-line and writing to file.
        img_rank_ascii.write(f'{os.linesep}'.join(pixels_row))

    return img_rank_file #(img_rank_file, image) # returning img-for-rank file, and pixels numpy array

# generates header.txt and svd.txt for given pgm file.
def generate_header_svd_files_for_pgm_image(filename):
    ascii_pgm_file_contents = img.read_ascii_pgm_file(filename)
    
    width, height, max_pixel = ascii_pgm_file_contents[:3]
    header_file , svd_file = utils.get_header_and_svd_txt_filenames(filename) 
    
    # generates header.txt file.
    create_header_text_file(header_file, width, height, max_pixel)

    # generating svd.txt file from pgm pixels data. 
    create_svd_text_file(svd_file, height, width, ascii_pgm_file_contents[3:])
    
# generates various ranks of images
# and produces tables, charts for the ranks.
def generate_images_by_rank_range(filename, *args):
    # fetching width, and height.
    width, height = img.read_ascii_pgm_file(filename)[:2]
    min_size = min(width, height)

    # generating header.txt and svd.txt files for the given image. 
    generate_header_svd_files_for_pgm_image(filename)
    
    # creates header.txt and SVD.txt file-names for given image.
    header_file , svd_file = utils.get_header_and_svd_txt_filenames(filename) 
    
    # sorting and filtering ranks passed.
    args = sorted([int(e) for e in args[0] if int(e) > 0 and int(e) <= min_size])
    if len(args) == 0:
        args = [e for e in range(min_size)] # creating all possible ranks otherwise.

    img_details = get_image_compression_analysis_details(filename, header_file, svd_file, args)
    np_img_details = np.array(img_details[1:], dtype=np.single)
    
    # writing svd conversion stats to csv file.
    file_without_extension = utils.remove_file_extension(filename)
    with open(f'{file_without_extension}_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(img_details)
    
    # plots graphs showing stats for given image and for given ranks.
    draw_graphs(np_img_details)

## ===== helper functions ====== ##

# creates header.txt file with passed image metadata.
def create_header_text_file(filename, width, height, max_pixel):    
    with open(filename, 'w') as header_txt_file:
        header_contents = f'{width} {height} {max_pixel}'
        header_txt_file.write(header_contents)
    print('created header.txt file, filename with relative path: ', filename)

# creates svd_txt file
def create_svd_text_file(svd_file, height, width, pixels_data):
    
    # pixel data to numpy array.
    np_pixels = np.array(pixels_data).reshape((height, width))

    U, S, V_t = np.linalg.svd(np_pixels, full_matrices = True)

    V = V_t.T # converting V-transpose to V

    # creating sigma matrix with (m, n) size.
    sigma = np.zeros((height, width), S.dtype)
    S = np.fill_diagonal(sigma, S)

    # writing data into svd_file
    with open(svd_file, 'w') as svd_txt:
        # saving U matrix
        for row in U:
            # print(row)
            np.savetxt(svd_txt, row, fmt="%.8g")

        # saving sigma(S) matrix             
        for row in sigma:
            # print(row)
            np.savetxt(svd_txt, row, fmt="%.8g")
        
        # saving V matrix
        for row in V:
            # print(row)
            np.savetxt(svd_txt, row, fmt="%.8g")
    
    print('created svd.txt file, path: ', svd_file)

# reads the header.txt file
def read_header_file(header_file):
    with open(header_file, 'r') as header_txt:
        width, height, max_pixel = [int(e) for e in header_txt.readline().split()]
    return (width, height, max_pixel)

# reads the svd.txt file contents.
# @returns the svd_data as 1D numpy array.
def read_svd_file(svd_file):
    with open(svd_file, 'r') as svd_txt_file:
        svd_data = svd_txt_file.readlines()
    
    # converting the svd data into linear list of numpy.single values.
    return [np.single(e) for line in svd_data for e in line.split()]

# reads svd compressed bin image.
# @returns the pgm-header, rank and pixel data as a tuple.
def read_svd_compressed_bin_image(bin_file):
    # using file open() to read file contents
    infile = open(bin_file, 'rb')
    header = next(infile)        
    metadata = next(infile)
    width, height, max_pixel, rank = [int(e) for e in metadata.split()]
    
    # checks if the file is in binary form.
    assert 'P5' in str(header) 

    # print(header, width, height, max_pixel, rank)
    pgm_header = f'P2{os.linesep}{width} {height}{os.linesep}{max_pixel}{os.linesep}'
    
    # file seek to the position: where pixel data starts.
    seek_pos = len(header) + len(metadata)
    infile.seek(seek_pos)

    # reading pixels in bytes to numpy array 
    image = np.fromfile(infile, dtype=np.single).reshape((height, width))

    # rounding the pixels to int and storing in numpy array
    image = np.array([round(pixel) for row in image for pixel in row]).reshape((height, width))

    # closing the reading bin_file.
    infile.close()
    return (pgm_header, rank, image)

# extracts U, S, V-transpose matrices from svd 1D numpy array.
def extract_svd_matrices(svd_data, height, width, k):

    min_size = min(height, width)
    
    # check if rank is valid, k is <= min(rows, columns)
    if (k > min_size):
        raise('invalid rank, rank <= min(rows, columns)')
    
    # size: a(m, n) => u(m, m), s(m, n), v(n, n)
    # m = rows | height, n = columns | width
    # NOTE: assuming sigma(S) matrix will be given with all elements.
    u_size, s_size, v_size = height*height, height*width, width*width

    # NOTE: if sigma(S) matrix is given with only diagonal elements, uncomment below line.            
    # u_size, s_size, v_size = height*height, min_size, width*width

    # extracting sublists based on the U|S|V sizes.
    u_tmp = svd_data[0:u_size]
    s_tmp = svd_data[u_size: (u_size + s_size)]
    v_tmp = svd_data[(u_size + s_size): (u_size + s_size + v_size)]

    U = np.array(u_tmp).reshape((height, height))
    # size of S will be: (1, min_size), if only considering diagonal elements
    S = np.array(s_tmp).reshape((height, width)) 
    V_t = np.array(v_tmp).reshape((width, width)).T 

    return (U, S, V_t)

# saves the file data passed into binary format with pgm.SVD extension
# @returns the created binary file name.
def save_compressed_pgm_file(filename, header_data, pixel_data):
    bin_file = filename + '_b.pgm.SVD'

    with open(bin_file, 'wb') as img_bin:
        width, height, max_pixel, rank = header_data
        
        # writing file header content to the file.
        pgm_header = f'P5{os.linesep}{width} {height} {max_pixel} {rank}{os.linesep}'
        img_bin.write(bytearray(pgm_header, encoding='utf-8'))
        
        # saving the pixel data row-by-row in binary format. 
        pixel_data.tofile(img_bin, sep='', format='%0.8g') 

    return bin_file

# calculates approximation error for svd for a given rank.
def calculate_approximation_error(height, width, S, k):
    if min(height, width) > k:
        return np.diag(S)[k] # k+1 element in S-diagonal
    else: 
        return 0

# get rate of compression
def get_rate_of_compression(original_size, appr_size):    
    return round((original_size - appr_size) * 100 / original_size, 2)

# gets the image compression analysis details.
def get_image_compression_analysis_details(filename, header_file, svd_file, args):

    # performs part-3 and 4 and calculates the appr. error, 
    # rate-of-compression for list of ranks.
    img_details = []    
    img_details.append(['Rank', 'Approximation Error', 'Rate of compression (ascii to approximated image)', 'Rate of compression (binary to approximated image)'])
    original_size = os.path.getsize(filename)
    
    for rank_k in args:
        rank_k_details = []
        # part-3: svd to rank-k approximated image in bin form
        bin_file, appr_error = svd_to_compressed_image(header_file, svd_file, rank_k)
        bin_file_size = os.path.getsize(bin_file)

        # part-4: bin. compressed image to approximated-image
        appr_image = convert_svd_compressed_bin_image_to_ascii(bin_file)
        appr_img_size = os.path.getsize(appr_image)

        rank_k_details.append(rank_k)
        rank_k_details.append(appr_error)

        # rate of compression for original to approx
        rc = get_rate_of_compression(original_size, appr_img_size)
        rank_k_details.append(rc)

        # rate of compression for bin to approx
        rc = get_rate_of_compression(bin_file_size, appr_img_size)
        rank_k_details.append(rc)

        img_details.append(rank_k_details)

    return img_details

# plots the graphs from given img_statistics array.
def draw_graphs(np_img_details):

    np_col_stack = np.column_stack(np_img_details[:])
    ranks, appr_errors, rc_org_appr, rc_bin_appr = np_col_stack
    
    # displays rate-of-compression compare graph
    plot1 = plt.figure(1)
    # plot two lines
    plt.plot(ranks, rc_org_appr, 'o-g')
    plt.plot(ranks, rc_bin_appr, 'o-b')
    
    # set axis titles
    plt.xlabel('Rank')
    plt.ylabel('Rate of compression')
    plt.title('"Rank" vs "Rate of compression"')
    
    # show label in legend in min-map.
    plt.legend(['orginal to approx image', 'bin to approx image'])
    # plt.show()
    
    # line-graph plot showing rank vs appr.error
    plot2 = plt.figure(2)
    plt.plot(ranks, appr_errors, 'o-g')
    plt.xlabel('Rank')
    plt.ylabel('Approximation Error')
    plt.title('"Rank" vs "Approximation error"')
    plt.show()
