"""
Created on Sat Apr 25 22:08:19 2020
@author: etshawy
"""
import math
import numpy as np
import cv2

img = cv2.imread('testdog.bmp')
height, width, channels = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = np.array(img)
flattened = data.flatten()

window_size = 123
lookahead_buffer_size = 20
search_buffer_size = window_size - lookahead_buffer_size

#flattened = np.array(['c', 'a', 'b', 'r', 'a', 'c', 'a', 'd',
#                   'a', 'b', 'r', 'a', 'r', 'r', 'a', 'r', 'r', 'a', 'd'])


def get_longest_match(stream, current_position):
    minimum_search_index = max(0, current_position - search_buffer_size - 1)
    maximum_search_index = min(
        len(flattened) - 1, current_position + lookahead_buffer_size - 1)
    search_buffer = stream[minimum_search_index: maximum_search_index + 1]
    lookahead_buffer = stream[current_position: maximum_search_index + 1]
    l = 2
    r = lookahead_buffer_size
    match_index = -1
    match_length = -1
    while l <= r: 
  
        mid = l + (r - l)//2; 
          
        # Check if x is present at mid 
        test_match_index, test_match_length = last_match_sequence(
            search_buffer, lookahead_buffer[: mid], minimum_search_index, current_position)
  
        # If x is greater, ignore left half 
        if test_match_index != -1: 
            l = mid + 1
            match_index = test_match_index
            match_length = test_match_length
  
        # If x is smaller, ignore right half 
        else: 
            r = mid - 1
        
    if match_index != -1:
        offset = current_position - match_index
        return (offset, match_length)

    current_position += 1
    return (0, 0)


def last_match_sequence(arr, seq, min_indx, current_pos):

    # Range of sequence
    r_seq = np.arange(seq.size)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(arr.size-seq.size+1)[:, None] + r_seq] == seq).all(1)
    matches_indeces = np.where(M == True)[0] + min_indx
    matching_indx = matches_indeces[matches_indeces < current_pos]

    # Get the range of those indices as final output
    if len(matching_indx) > 0:
        return (matching_indx[-1], len(seq))
    else:
        return (-1, -1)         # No match found


current_position = 0

print('encoding...')
codes = []
flags = []
pixels = []
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

codes = np.array(codes, dtype=np.uint16)
pixels = np.array(pixels, dtype=np.uint8)
flags_before = np.array(flags, dtype=np.bool)
flags = np.packbits(flags_before, axis=None)
# save the binary data for the codes
np.save('image.npy', pixels)
np.save('prefixes.npy', codes)
np.save('flags.npy', flags)
print(len(pixels))
print(len(flattened))
print('decoding...')
flags_after = np.unpackbits(flags, axis=None).astype(np.bool)

decoded = []
tag = 0
pixels_index = 0
codes_index = 0
while pixels_index < len(pixels) and tag < len(flags_after):
    if flags_after[tag] == False:
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
    tag += 1

to_remove = len(decoded) % (height * width)
if to_remove:
    decoded = decoded[:-to_remove]

# turn the flat decoded blocks into an image providing its width and height
decoded_img = np.reshape(decoded, (height, width))
decoded_img = decoded_img.astype(np.uint8)

# show the decoded image and store it as output.png
cv2.imshow('Gray image', decoded_img)
cv2.imwrite('testdog.bmp', decoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()