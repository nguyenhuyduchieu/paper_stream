import cv2
import numpy as np
import utlis
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time



########################################################################
webCamFeed = True
pathImage = "/Users/hieuduc/Downloads/365533666_102049012996215_550280897591197067_n.png"
#cap = cv2.VideoCapture(0)
#cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################

utlis.initializeTrackbars()
count=0


while True:
    start_time = time.perf_counter()
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
 
    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5) # DRAW ALL DETECTED CONTOURS
    x1 = contours[0].shape[0]
    reshaped_contours = np.reshape(contours[0],(x1,2))
    x_point = sorted(reshaped_contours, key = lambda x:x[0])
    x_min = x_point[0]
    x_max = x_point[-1]
    y_point = sorted(reshaped_contours, key = lambda x:x[1])
    y_min = y_point[0]
    y_max = y_point[-1]

 
    # FIND THE BIGGEST COUNTOUR
    # biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    # if biggest.size != 0:
    #     biggest=utlis.reorder(biggest)
    #     print(biggest)
    #     cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    #     imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
    #     pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    #     pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    #     matrix = cv2.getPerspectiveTransform(pts1, pts2)
    #     imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
 
    #     #REMOVE 20 PIXELS FORM EACH SIDE
    #     imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    #     imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
 
    #     # APPLY ADAPTIVE THRESHOLD
    #     # imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    #     # imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    #     # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    #     # imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
 
    #     # Image Array for Display
    #     imageArray = ([img,imgThreshold,imgContours],
    #                   [imgBigContour,imgWarpColored])
        
    #     cv2.imshow("perspectiveGray",imgWarpColored)
    #     cv2.imshow("bigContour", imgBigContour)
    #     end_time = time.perf_counter()
    #     print("The total time taken is:", end_time - start_time)
    #     cv2.waitKey(0)
 
    # else:
    # input_pts = np.float32([(x_min[0], y_min[1]), (x_min[0], y_max[1]),(x_max[0], y_max[1]),(x_max[0], y_min[1])])
    input_pts = np.float32([y_min, y_max, x_min, x_max])
    output_pts = np.float32([[0, 0],
                            [widthImg, heightImg],
                            [widthImg, 0],
                            [0, heightImg]])
    new_biggest = np.array([x_min,
                            y_max,
                            y_min,
                            x_max])
    new_biggest = new_biggest.reshape(4, 1, 2)
    cv2.drawContours(imgBigContour, new_biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    imgBigContour = utlis.drawRectangle(imgBigContour,new_biggest,2)
    matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 10, 20:imgWarpColored.shape[1] - 10]
    imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    # imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    # imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
    
    imageArray = ([img,imgThreshold,imgContours],
                    [imgBigContour, imgWarpColored, imgWarpGray, imgBlank])
    
    cv2.imshow("bigContour", imgBigContour)
    cv2.imshow("perspectiveColor",imgWarpColored)

    end_time = time.perf_counter()
    print("The total time taken is:", end_time - start_time)
    cv2.waitKey(0)
