'''importing libraries'''

import cv2
import numpy as np

'''importing a file will make it readable,matrix'''


def read_img(project):
    img = cv2.imread(project)
    return img
 
'''edge detection'''

def  edge_detection(img,line_wdt,blur):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayblur = cv2.medianBlur(gray,blur)
    edges = cv2.adaptiveThreshold(grayblur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_wdt,blur)
    return edges

'''will reduce the color the image will look like a cartoon'''

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret, label, center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

'''add the color image and read the image'''

img = read_img('./img2.jpg')
line_wdt = 9
blur_value = 7
totalColors = 4


edgeImg = edge_detection(img,line_wdt,blur_value)
img = color_quantisation(img,totalColors)
blurred = cv2.bilateralFilter(img, d=7,sigmaColor=200,sigmaSpace=200)
cartoon = cv2.bitwise_and(blurred,blurred,mask=edgeImg)

'''show the img in form of normal img,edge img and cartoon img'''

cv2.imshow('Image',img)
cv2.imshow('cartoon',cartoon)
cv2.imshow('edges',edgeImg)
cv2.waitKey(0)
cv2.destroyAllWindows()