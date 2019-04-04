import sys
import argparse
import cv2
#使用opencv2的接口，帮助我们处理视频
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        print ('Read a new frame: ', success)
        #视频的每5帧，存一张jpg图片
        if count % 5 == 0: 
            cv2.imwrite( pathOut + "/frame%05d.jpg" % count, image)     
        count = count + 1
        success,image = vidcap.read()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    #输入视频地址
    a.add_argument("--input", help="path to video")
    #输出图片目录
    a.add_argument("--output", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.input, args.output)
