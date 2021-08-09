# import the necessary packages
from shape import ShapeDetector
import argparse
import imutils
import cv2
import time
# construct the argument parse and parse the arguments
from imutils.video import VideoStream,FileVideoStream
import numpy as np
from scipy.spatial import distance

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
# ap.add_argument("-p", "--picamera", type=int, default=-1,
# 	help="whether or not the Raspberry Pi camera should be used")
# ap.add_argument("-f", "--fps", type=int, default=30,
# 	help="FPS of output video")
# ap.add_argument("-c", "--codec", type=str, default="MJPG",
# 	help="codec of output video")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera
# sensor to warmup
#print("[INFO] warming up camera...")

vs = cv2.VideoCapture('./saved_videos/depth_new.avi') ### change the video processor
time.sleep(2.0)

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = None
(h, w) = (None, None)
zeros = None

# object_detector = cv2.createBackgroundSubtractorMOG2()


try:
    while True:
        # grab the frame from the video stream and resize it to have a
        # maximum width of 300 pixels
        # print("Reading the video...")
        ret, frame = vs.read()

        #check if the writer is None
        if writer is None:
            # store the image dimensions, initialize the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter(args["output"], fourcc, 30,(w , h), True)

        # Select ROI
        upper_left = (300, 0)
        bottom_right = (390, 300)

        # Crop image
        new_frame_blue = frame.copy()
        new_frame_green = frame.copy()
        
        #find center of image and draw it (blue circle)
        image_center_x = int((bottom_right[0]+upper_left[0])/2)
        image_center_y = int((bottom_right[1]+upper_left[1])/2)
        image_center = (image_center_x,image_center_y)


        ####################                 BLUE                  ####################
        # Set minimum and max HSV values to display
        lower_blue = np.array([90, 0, 0])
        upper_blue = np.array([120, 255, 255])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(new_frame_blue, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        new_frame_blue = cv2.bitwise_and(new_frame_blue,new_frame_blue, mask= mask)
        ################################################################################

        #Rectangle marker
        r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 2)

        
        # ####################                 GREEN                  ####################

        # lower_green = np.array([60, 0, 0])
        # upper_green = np.array([90, 255, 255])

        # # Create HSV Image and threshold into a range.
        # hsv = cv2.cvtColor(new_frame_green, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # new_frame_green = cv2.bitwise_and(new_frame_green, new_frame_green, mask= mask)
        # ################################################################################

        gray = cv2.cvtColor(new_frame_blue, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred = cv2.bilateralFilter(gray,9,75,75)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the shape detector
        cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        buildings = []

        for cnt in cnts:
            M = cv2.moments(cnt)
            center_X = int((M["m10"] / (M["m00"]+ 1e-5)))
            center_Y = int((M["m01"] / (M["m00"]+ 1e-5)))
            contour_center = (center_X, center_Y)

            area = cv2.contourArea(cnt)
            # if area > 50:

            # calculate distance to image_center
            distances_to_center = (distance.euclidean(image_center, contour_center))
    
            # save to a list of dictionaries
            buildings.append({'contour': cnt, 'center': contour_center, 'distance_to_center': distances_to_center, 'Area': area})
    

            # area = cv2.contourArea(cnt)
            # if area > 50:
            #cv2.drawContours(new_frame, [cnt], 0, (255, 255, 255), 2)

        
        cv2.circle(frame, image_center, 3, (0, 0, 255), 2)

        # sort the buildings
        sorted_buildings = sorted(buildings, key=lambda i: i['distance_to_center'])
        
        if sorted_buildings[0]['distance_to_center'] < 50 and sorted_buildings[0]['Area']>10:
            # find contour of closest building to center and draw it (blue)
            center_building_contour = sorted_buildings[0]['contour']
            x, y, w, h = cv2.boundingRect(center_building_contour)
            cv2.putText(frame, 'Area: '+str(sorted_buildings[0]['Area']), (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.drawContours(frame, [center_building_contour], 0, (255, 255, 255), 2)
    
          

        writer.write(frame)
        # show the output image
        cv2.imshow("Shapes", frame)
        cv2.imshow("Thresh", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except Exception as ex:
    print("There is something wrong with video processing!! TRY a PERMANENT FIX!!!")
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)

finally:
    # do a bit of cleanup
    print("[INFO] cleaning up...")
    vs.release()
    print("Video capture release!!")
    writer.release()
    print("Writer release!!")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("All done!!")