# puffins-and-penguins-official-final-project-

#!/usr/bin/python3

import jetson_inference
import jetson_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Atlantic_Puffin.jpg", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

# load an image (into shared CPU/GPU memory)
img = jetson_utils.loadImage(opt.Atlantic_Puffin.jpg)

# load the recognition network
net = jetson_inference.imageNet(opt.network)

# classify the image
class_idx, confidence = net.Classify(img)

# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")
