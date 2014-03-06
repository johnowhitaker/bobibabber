import numpy as np

def moveHaarWindow(im, wwidth, wheight, xincrement, yincrement, v):
    features = []
    imheight = len(im)
    imwidth = len(im[0])
    wx, wy = 0, 0
    while wy < (imheight-wheight):
        wx = 0
        while wx < (imwidth-wwidth):
            dark = 0
            bright = 0
            for y in range(wy, (wy+wheight)):
                for x in range(wx, wx+wwidth):
                    luminance = im[x][y]
                    if v:                                   #vertical box
                        if y < (wy+wheight/2):
                            dark += luminance
                        else:
                               bright += luminance
                        
                    else:
                        if x < (wx+wwidth/2):
                            dark += luminance
                        else:
                               bright += luminance
            result = (bright-dark+float(wwidth*wheight)/2*255)/(wwidth*wheight*255)
            features.append(round(result, 8))
            wx += xincrement
        wy += yincrement
        return features

def getHaarBox(im,x,y,w,h, v):
    features = []
    imheight = len(im)
    imwidth = len(im[0])
    wx, wy = x, y
    dark = 0
    bright = 0
    for y in range(wy, (wy+wheight)):
        for x in range(wx, wx+wwidth):
            luminance = im[x][y]
            if v:                                   #vertical box
                if y < (wy+wheight/2):
                    dark += luminance
                else:
                    bright += luminance
                
            else:                                   #horizantal
                if x < (wx+wwidth/2):
                    dark += luminance
                else:
                    bright += luminance
        result = (bright-dark+float(wwidth*wheight)/2*255)/(wwidth*wheight*255)
        features.append(round(result, 8))
        wx += xincrement
    wy += yincrement
    return features
    
def getHaarFeatures(im):
    #not bad - 94% at 500 iterations
    a = moveHaarWindow(im, 12, 6, 2, 2, True)
    b = moveHaarWindow(im, 10, 20, 3, 3, True)
    c = moveHaarWindow(im, 6, 15, 2, 5, False)
    d = moveHaarWindow(im, 10, 5, 2, 5, False)
    features = a + b + c + d
    return features
    

def getFeatures1(im):
    #very fast, but only 88% at 500 iterations (its only 24 features)
    a = moveHaarWindow(im, 20, 20, 5, 5, True)
    b = moveHaarWindow(im, 20, 20, 5, 5, False)
    c = moveHaarWindow(im, 8, 15, 2, 2, False)
    features = a + b + c
    return features

