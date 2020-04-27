# Gray scale images LZ77 Coding

## prerequisites

python `3.7`

## how to run

1. run the file `main.py`

2. you will be prompted for the path of the image, enter the path without quotes

3. enter the window size used for encoding

4. enter the size of the look ahead buffer

5. you will find encoded input in files `image.npy`, `flags.npy`, `prefixes.npy` 
    * the first one contains any pixel code that needs to be stored
    * second one contains flag that comes before any tag to indicate if length and offset are stored along with the pixel code or the pixel code only stored. this is made as an optimization step.
    * last one contains the offset the length of matching of tags

6. you will find decoded output in `output.png`

