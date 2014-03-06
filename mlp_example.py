from sklearn.datasets import load_digits
from multilayer_perceptron  import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor
import numpy as np
from matplotlib import pyplot as plt
import glob
import Image
import datetime
from Haar_extractor import *
import warnings, os, sys
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") #removes the deprecation warnings

print "Start:",datetime.datetime.now()

def loadImage(imagepath):
    #load the image and convert to an array
    im=Image.open(imagepath)
    pixels = list(im.getdata())
    final_pix = []
    for pix in pixels:
        p = (pix[0]+pix[1]+pix[2])/3
        final_pix.append(p)
    width, height = im.size
    pixels = [final_pix[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels
    
def loadSubImage(im):
    pixels = list(im.getdata())
    final_pix = []
    for pix in pixels:
        p = (pix[0]+pix[1]+pix[2])/3
        final_pix.append(p)
    width, height = im.size
    pixels = [final_pix[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels

#load images and get first features
pat = []
paty = []
test = []
testy = []
positive_images = glob.glob("/home/jonathan/Baobab/fromGE/train/baobab/poss/pos*.png")
for im in positive_images:
    image = loadImage(im)
    haars = getFeatures1(image)
    pat.append(haars)
    paty.append(1)
negative_images = glob.glob("/home/jonathan/Baobab/fromGE/train/wbackground/bg*.png")
for im in negative_images:
    image = loadImage(im)
    haars = getFeatures1(image)
    pat.append(haars)
    paty.append(0)
test_images = glob.glob("/home/jonathan/Baobab/fromGE/train/test/*.png")
for im in test_images:
    image = loadImage(im)
    haars = getFeatures1(image)
    test.append(haars)
X = np.array(pat)
y = np.array(paty)

print "Finished loading images and computing first feature set:",datetime.datetime.now()

pat = []
paty = []
positive_images = glob.glob("/home/jonathan/Baobab/fromGE/train/baobab/poss/pos*.png")
p = 1
for im in positive_images:
    p += 1
    image = loadImage(im)
    haars = getHaarFeatures(image)
    pat.append(haars)
    paty.append(1)
negative_images = glob.glob("/home/jonathan/Baobab/fromGE/train/wbackground/bg*.png")
for im in negative_images:
    image = loadImage(im)
    haars = getHaarFeatures(image)
    pat.append(haars)
    paty.append(0)
X2 = np.array(pat)
y2 = np.array(paty)

print "Finished loading images and computing second feature set:",datetime.datetime.now()


print "Beginning Training of first classifier:",datetime.datetime.now()

# MLP training performance
mlp = MultilayerPerceptronClassifier(n_hidden = 5,max_iter = 500, alpha = 0.02)
mlp.fit(X, y)
print "Beginning Training of second classifier:",datetime.datetime.now()
mlp2 = MultilayerPerceptronClassifier(n_hidden = 5,max_iter = 800, alpha = 0.02)
mlp2.fit(X2, y2)
print "Finished Training:",datetime.datetime.now()

print "Training Score = ", mlp.score(X,  y)
print "Training Score 2", mlp2.score(X2, y2)
#print "Predicted labels = ", mlp.predict(X)
#print "True labels = ", y
#print(datetime.datetime.now())
#print "Pedicted for test:", mlp.predict(test)
#print "Pedicted for test:", mlp.predict_proba(test)
#print(datetime.datetime.now())

#~ #test on a real image
print "Beginning Test of image:",datetime.datetime.now()
test = []
im_test=Image.open("/home/jonathan/Baobab/fromGE/256.png")
box_width = 40
box_height = 40

imheight = 700 #GET FROM IMAGE!!!
imwidth = 1200
x, y = 0, 0
baobabs = []
while y < (imheight-box_height):
	x = 0
	while x < (imwidth - box_width):
		region = im_test.crop((x,y,x+box_width, y+box_width))
		small_image = loadSubImage(region)
		haars = getFeatures1(small_image)
		test.append(haars)
		results = mlp.predict(test)
		test = []
		if results[0] == 1:
			baobabs.append((x,y))
		x += 10
	y += 10
print "narrowed down from ~8000 to",len(baobabs)
print "starting second round:",datetime.datetime.now()
baobabs2 = []
#Fix This, need to train a new net with getHaarFeatures...
for b in baobabs:
	x,y = b
	region = im_test.crop((x,y,x+box_width, y+box_width))
	small_image = loadSubImage(region)
	haars = getHaarFeatures(small_image)
	test.append(haars)
	results = mlp2.predict(test)
	test = []
	if results[0] == 1:
		baobabs2.append((x,y))
print "narrowed down from %i to" % len(baobabs),len(baobabs2)
print "Finished:",datetime.datetime.now()
print baobabs2

	


