"""
Created on Sat Apr 25 22:08:19 2020
@author: etshawy
"""
import math
import numpy as np
import cv2

img = cv2.imread('baboon.bmp')
height, width, channels = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = np.array(img)
flattened = data.flatten().tolist()

window_size = 100
lookahead_buffer_size = 50
search_buffer_size = window_size - lookahead_buffer_size

#flattened = ['c', 'a', 'b', 'r', 'a', 'c', 'a', 'd',
#                      'a', 'b', 'r', 'a', 'r', 'r', 'a', 'r', 'r', 'a', 'd']


def get_longest_match(data, current_position):
    """ 
    Finds the longest match to a substring starting at the current_position 
    in the lookahead buffer from the history window
    """
    end_of_buffer = min(current_position +
                        lookahead_buffer_size, len(data) + 1)

    best_match_distance = -1
    best_match_length = -1

    # Optimization: Only consider substrings of length 2 and greater, and just
    # output any substring of length 1 (8 bits uncompressed is better than 13 bits
    # for the flag, distance, and length)
    for j in range(current_position + 2, end_of_buffer):

        start_index = max(0, current_position - window_size)
        substring = data[current_position:j]

        for i in range(start_index, current_position):

            repetitions = int(len(substring) / (current_position - i))

            last = len(substring) % (current_position - i)

            matched_string = data[i:current_position] * \
                repetitions + data[i:i+last]

            if matched_string == substring and len(substring) > best_match_length:
                best_match_distance = current_position - i
                best_match_length = len(substring)

    if best_match_distance > 0 and best_match_length > 0:
        return (best_match_distance, best_match_length)
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

codes = []
print('encoding...')
while current_position < len(flattened):
    offset, length = get_longest_match(flattened, current_position)
    current_position += length 
    next_code = flattened[current_position]
    current_position += 1
    codes.append([offset, length, next_code])

tags = np.array(codes, dtype=np.uint8)

# save the binary data for the codes
np.save('image.npy', tags)

print('decoding...')
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
