"""
Created on Sat Apr 25 22:08:19 2020
@author: etshawy
"""
import math
import numpy as np
import cv2

img = cv2.imread('test.png')
height, width, channels = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = np.array(img)
flattened = data.flatten()

window_size = 256
lookahead_buffer_size = 128
search_buffer_size = window_size - lookahead_buffer_size

# string = np.array(['c', 'a', 'b', 'r', 'a', 'c', 'a', 'd',
#                   'a', 'b', 'r', 'a', 'r', 'r', 'a', 'r', 'r', 'a', 'd'])


def get_longest_match(stream, current_position):
    minimum_search_index = max(0, current_position - search_buffer_size - 1)
    maximum_search_index = min(
        len(flattened) - 1, current_position + lookahead_buffer_size - 1)
    search_buffer = stream[minimum_search_index: maximum_search_index + 1]
    lookahead_buffer = stream[current_position: maximum_search_index + 1]
    for seq_length in range(lookahead_buffer_size, 0, -1):
        match_index, match_length = last_match_sequence(
            search_buffer, lookahead_buffer[: seq_length], minimum_search_index, current_position)
        if match_index != -1:
            offset = current_position - match_index
            current_position += (match_length + 1)
            return (offset, match_length, stream[current_position - 1], current_position)

    current_position += 1
    return (0, 0, stream[current_position - 1], current_position)


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

codes = []
print(codes)
while current_position < len(flattened):
    offset, length, next_code, current_position = get_longest_match(
        flattened, current_position)
    codes.append([offset, length, next_code])

tags = np.array(codes)
print(codes)
print(tags)
# save the binary data for the codes
#    np.save('image.npy', tags)

decoded = []
for tag in tags:
    if tag[0] == 0:
        decoded.append(tag[2])
    else:
        sequence_index = len(decoded) - tag[0]
        for letter in range(tag[1]):
            decoded.append(decoded[sequence_index])
            sequence_index += 1
        decoded.append(tag[2])


# turn the flat decoded blocks into an image providing its width and height
decoded_img = np.reshape(flattened, (height, width))
decoded_img = decoded_img.astype(np.uint8)

# show the decoded image and store it as output.png
cv2.imshow('Gray image', decoded_img)
cv2.imwrite('output.png', decoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
