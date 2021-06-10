import cv2
import os

VIDEO_DIR = "videos"
PICTURE_DIR = "pictures"

for f in os.listdir(VIDEO_DIR):
    os.mkdir(PICTURE_DIR + '/' + f[:-4])
    vidcap = cv2.VideoCapture(VIDEO_DIR + '/' + f)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(PICTURE_DIR + '/' + f[:-4] + '/' + "frame%d.jpg" % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        for i in range(0, 9):
            success, image = vidcap.read()
        count += 10
