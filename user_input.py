#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:50:56 2020

@author: etshawy
"""
import  cv2

def input_int(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
       break 

def read_gray_img():
    while True:
        path = input('please enter the absolute path to the image: ')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if not img.any():
            print("please provide a right path to image, terminating...")
        else:
            return img
