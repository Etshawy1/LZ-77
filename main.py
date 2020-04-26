"""
Created on Sat Apr 25 22:08:19 2020
@author: etshawy
"""
import math
import numpy as np
import cv2
import more_itertools
import user_input

min_acceptable_sequence_length = 2
data_type = 'B'
img = user_input.read_gray_img()
height, width = img.shape
window_size = user_input.input_int('please enter the window size: ')
if window_size > 255:
    data_type = 'uint16' # data type of the prefixes of the tags (offset, length)
lookahead_buffer_size = user_input.input_int('please enter the look ahead buffer size: ')
if lookahead_buffer_size > window_size:
    print("buffer can't be greater than the whole window, terminating program...")
    exit(-1)

flattened = img.flatten().tolist()
search_buffer_size = window_size - lookahead_buffer_size

def get_longest_match(data, current_position):
    """Summary or Description of the Function

    Parameters:
    data (array of int): the flattened image pixel values
    current_position (int): the index of the start of the lookahead buffer

    Returns:
    tuple : (offset of the longest match from the start of the lookahead buffer, length of the longest match) or (0, 0) if not any

   """

    minimum_search_index = max(0, current_position - search_buffer_size - 1)
    maximum_search_index = min(len(flattened) - 1, current_position + lookahead_buffer_size - 1)
    search_buffer = data[minimum_search_index: maximum_search_index + 1]
    lookahead_buffer = data[current_position: maximum_search_index + 1]
    l = min_acceptable_sequence_length
    r = lookahead_buffer_size
    match_index = -1
    match_length = -1
    while l <= r:
        mid = l + (r - l)//2
        # get match index of legth = mid
        test_match_index = match_sequence(search_buffer, lookahead_buffer[: mid], minimum_search_index, current_position)
        # there is a match with length = mid so test if there is higher length
        if test_match_index != -1:  
            l = mid + 1
            match_index = test_match_index
            match_length = mid
        # check for lower match lengths
        else: 
            r = mid - 1

    if match_index != -1:
        offset = current_position - match_index
        return (offset, match_length)
    return (0, 0)


def match_sequence(arr, sequence, min_indx, current_pos):
    possible_matches = list(more_itertools.windowed(arr, len(sequence)))
    for sequence_start_index in range(current_pos - min_indx + 1):
        if sequence_start_index + min_indx >= current_pos:
            return -1
        if list(possible_matches[sequence_start_index]) == sequence:
            return (sequence_start_index + min_indx)
    return (-1)



current_position = 0
codes = []  # store the offset and length of each tag
flags = []  # flag to indicate if next is a tag or just a charachter code
pixels = [] # pixel values in tags or without tags depends on the corresponding flag
print('encoding...')
while current_position < len(flattened):
    offset, length = get_longest_match(flattened, current_position)
    current_position += length
    if current_position < len(flattened):
        next_code = flattened[current_position]
    else:
        next_code = -1
    current_position += 1
    if length == 0:
        pixels.append(next_code)
        flags.append(False)
    else:
        codes.append(offset)
        codes.append(length)
        pixels.append(next_code)
        flags.append(True)


codes = np.array(codes, dtype=data_type)
pixels = np.array(pixels, dtype=np.uint8)
flags_before = np.array(flags, dtype=np.bool)
flags = np.packbits(flags_before, axis=None)

# save the binary data for the codes
np.save('image.npy', pixels)
np.save('prefixes.npy', codes)
np.save('flags.npy', flags)

# printed to check the performace of the compression
#print(len(flattened))
#print(len(flags_before))


print('decoding...')

# read the necessary files for decoding
flags = np.load('flags.npy')
codes = np.load('prefixes.npy')
pixels = np.load('image.npy')

flags_after_unpacking = np.unpackbits(flags, axis=None).astype(np.bool)
decoded = []
tag_index = 0
pixels_index = 0
codes_index = 0
while pixels_index < len(pixels) and tag_index < len(flags_after_unpacking):
    if flags_after_unpacking[tag_index] == False:
        decoded.append(pixels[pixels_index])
        pixels_index += 1
    else:
        sequence_index = len(decoded) - codes[codes_index]
        codes_index += 1
        length = codes[codes_index]
        codes_index += 1
        for letter in range(length):
            decoded.append(decoded[sequence_index])
            sequence_index += 1
        decoded.append(pixels[pixels_index])
        pixels_index += 1
    tag_index += 1


# remove any padding from the flattened image
to_remove = len(decoded) % (height * width)
if to_remove:
    decoded = decoded[:-to_remove]

# turn the flat decoded image into an image providing its width and height
decoded_img = np.reshape(decoded, (height, width))
decoded_img = decoded_img.astype(np.uint8)

# show the decoded image and store it as output.png
cv2.imshow('Gray image', decoded_img)
cv2.imwrite('output.png', decoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
