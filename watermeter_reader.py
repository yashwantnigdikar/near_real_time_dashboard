# Steps: 1. Read the image file
# 2. convert image to grayscale and resize it
# 3. Detect corners of the area of the interest
# 4. Transform corners to form the image
# 5. Segment the image based on the distance from the corners
# 6. RUN the OCR on the different segments and detect text and read it

import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from utils import stackImages
import matplotlib.pyplot as plt
import pytesseract
import easyocr

# path to tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)

    max_area = np.max(areas)
    for i, area in enumerate(areas):
        cv2.drawContours(imgContour, contours[i], -1, (255, 0, 0), 3)
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.1 * peri, True)
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)

    return imgContour  ## Corners of the shape


def cropSelection(img, coordinates):
    x, y, w, h = coordinates
    crop_img = img[y:y + h, x:x + w]
    return crop_img


def sortingCorners(corners):
    """
    :param corners: corners of the rectangle
    :return: sorted corners of the rectangle in the clockwise manner
    """
    # sort the points based on their x-coordinates
    xSorted = corners[np.argsort(corners[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def transformSelection(img, corners):
    width, height = 600, 400
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput

def binaryConvert(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return thres

# def cropThres(img):


def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgThres = cv2.erode(imgDil, kernel, iterations=1)
    return imgThres


# def cutScreen(img, pts):
#     return img[:pts[0], :,:]

def read_files(path):
    # we shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))

    return filelist


file = 'test3.jpeg'


img = cv2.imread(file)

# imgProcessed = preprocess(img)
# # cv2.imshow('processed', imgProcessed)
# # cv2.waitKey(0)
# #
# # imgContour = img.copy()
# # imgContour = getContours(imgProcessed, imgContour)
# # imgTransformed = transformSelection(img, corners)

img_cut = img[537:645, 80:416, :]
cv2.imshow('img',img_cut)
cv2.waitKey(0)
reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(img_cut)
print(result[0][1])
#
# pts = [440, 0, 440, 395]
# imgTransformed = cutScreen(imgTransformed, pts)

# imgTransformed = imgTransformed[:, :426, :]
# R = imgTransformed[15:115, 13:, :]
# Q = imgTransformed[130:220, 13:310, :]
# F = imgTransformed[245:318, 13:283, :]
# V = imgTransformed[305:355, 13:260, :]
#
# sliced = [binaryConvert(im) for im in [R, Q, F, V]]
# stacked = stackImages(0.5, [R, Q, F, V])
#
# for slice in sliced[:2]:
#     print(pytesseract.image_to_string(slice))
#     # cv2.imshow('img_bin', slice)
#     # cv2.waitKey()
#
# for slice in sliced[2:]:
#     reader = easyocr.Reader(['en'], gpu=True)
#     result = reader.readtext(slice)
#     print(result[0][1])
#
# # cv2.imshow('Img original', img)
# cv2.imshow('Image', stacked)
# cv2.waitKey(0)
