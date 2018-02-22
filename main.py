# import the necessary packages
from scipy.spatial import distance as dist
# import matplotlib.pyplot as plt
import numpy as np
# import argparse
# import glob
import subprocess
import json
import cv2
import thread

# start capturing video from cam
# cap = cv2.VideoCapture(0)

# start capturing video from other video file
cap = cv2.VideoCapture("videoplayback.mp4")
time_file = open("test_result/frame_time.txt","w")
# key-frame counter
counter = 0
previous_frame = ""
previous_hist = ""

# CAP_PROP_PAN = 33
# CAP_PROP_POS_AVI_RATIO = 2
# CAP_PROP_POS_FRAMES = 1
# CAP_PROP_POS_MSEC = 0
# CAP_PROP_PVAPI_BINNINGX = 304
# CAP_PROP_PVAPI_BINNINGY = 305
# CAP_PROP_PVAPI_DECIMATIONHORIZONTAL = 302

def check_for_number_plate (image_path, frame_time, previous_frame, cc):
    cmd = "alpr -c us " + image_path + " -j"
    alpr_result = subprocess.check_output(['bash', '-c', cmd])
    alpr_result = json.loads(alpr_result)
    if len(alpr_result["results"]) > 0:
        plate_no = alpr_result["results"][0]["plate"]
        coordinates = alpr_result["results"][0]["coordinates"]
        time_file.write("%18s | %8s | %102s | %fms\n" % ("key_frame_"+str(cc)+".jpg", plate_no, str(coordinates), frame_time))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # write the flipped frame
        # out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    	# extract a 3D RGB color histogram from the image,
    	# using 8 bins per channel, normalize, and update
    	# the index
    	hist = cv2.calcHist([current_frame], [0, 1, 2], None, [8, 8, 8],
    		[0, 256, 0, 256, 0, 256])
    	current_hist = cv2.normalize(hist).flatten()

        # METHOD #1: UTILIZING OPENCV
        # initialize OpenCV methods for histogram comparison
        # methodName = "Chi-Squared"
        # method = cv2.cv.CV_COMP_CHISQR
        methodName = "Manhattan"
        method = dist.cityblock
        methodName2 = "Chebysev"
        method2 = dist.chebyshev

        if counter == 0:
            previous_frame = frame
            previous_hist = current_hist
            cv2.imwrite("test_result/key_frame_"+str(counter)+".jpg", previous_frame)
            thread.start_new_thread( check_for_number_plate, ("test_result/key_frame_" + str(counter) + ".jpg", cap.get(0), previous_frame, counter) )
            counter += 1
        else:
            # result = cv2.compareHist(previous_hist, current_hist, method)
            result = method(previous_hist, current_hist)
            result2 = method2(previous_hist, current_hist)
            # print(result, result2)
            # result *= 5
            if result > 0.4 and result2 > 0.2:
                previous_frame = frame
                previous_hist = current_hist
                cv2.imwrite("test_result/key_frame_" + str(counter) + ".jpg", previous_frame)
                thread.start_new_thread( check_for_number_plate, ("test_result/key_frame_" + str(counter) + ".jpg", cap.get(0), previous_frame, counter) )
                counter += 1

        # bash_command = "alpr h786poj.jpg"
        # output = subprocess.check_output(['bash', '-c', bash_command])
        # output, error = subprocess.communicate()
        # print(output)

        # results = alpr.recognize_file("img.jpg")
        # i = 0
        #
        # for plate in results['results']:
        #     i += 1
        #     print("Plate #%d" % i)
        #     print("   %12s %12s" % ("Plate", "Confidence"))
        #     for candidate in plate['candidates']:
        #         prefix = "-"
        #         if candidate['matches_template']:
        #             prefix = "*"
        #
        #         print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
    else:
        break

# Release everything if job is finished
cap.release()
# out.release()
# alpr.unload()
cv2.destroyAllWindows()
