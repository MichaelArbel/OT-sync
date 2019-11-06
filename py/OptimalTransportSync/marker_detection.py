import numpy as np
import cv2
import cv2.aruco as aruco

#aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
print(aruco_dict)
# second parameter is id number
# last parameter is total image size
img = aruco.drawMarker(aruco_dict, 2, 700)
cv2.imwrite("d:/Data/test_marker.jpg", img)

frame = cv2.imread("D:/Data/marker/1_images/IMG_20160531_095403.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
parameters = aruco.DetectorParameters_create()

# print(parameters)

'''    detectMarkers(...)
    detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
    mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''
# lists of ids and the corners beloning to each id
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
print(corners)

# It's working.
# my problem was that the cellphone put black all around it. The alrogithm
# depends very much upon finding rectangular black blobs

gray = aruco.drawDetectedMarkers(gray, corners)

# print(rejectedImgPoints)
# Display the resulting frame
cv2.imshow('frame', gray)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()