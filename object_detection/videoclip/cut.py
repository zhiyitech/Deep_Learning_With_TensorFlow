import sys
import argparse
import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        print ('Read a new frame: ', success)
        if count % 5 == 0: #save jpg every 5 frames
            cv2.imwrite( pathOut + "/frame%05d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1
        success,image = vidcap.read()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.input, args.output)
