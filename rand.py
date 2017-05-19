#this method has two parameters, one is the number of labels retained, and the second is the number of images from each label

import random
import numpy as np
import sys, os
import argparse
from scipy.misc import imsave
import shutil

parser = argparse.ArgumentParser(description='Randomizes accross the number of labels and the number of images in each label')

labels = os.listdir('/Users/bs9sc/Desktop/101_ObjectCategories')
os.makedirs('/Users/bs9sc/Desktop/102_ObjectCategories')

parser.add_argument("-l", "--label", default = len(labels), type =int, help = "The number of labels we get")
parser.add_argument("-m", "--images", default = 0, type =int, help = "The number of images we get from each label")
args = parser.parse_args()
finalLabels = random.sample(labels, args.label)

print finalLabels

for label in finalLabels:
    
    
    myLabel = os.listdir('/Users/bs9sc/Desktop/101_ObjectCategories/' + str(label))
    finalImages = random.sample(myLabel, args.images)
        
    print finalImages
    os.makedirs('/Users/bs9sc/Desktop/102_ObjectCategories/' + str(label))
    
    for img in finalImages:
        shutil.copy('/Users/bs9sc/Desktop/101_ObjectCategories/' + str(label) + '/'+ str(img), '/Users/bs9sc/Desktop/102_ObjectCategories/' + str(label)+'/'+str(img))