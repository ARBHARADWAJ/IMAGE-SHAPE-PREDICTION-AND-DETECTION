import cv2
import numpy as np
import os
# from gtts import gTTS
from tkinter import *
import tkinter.messagebox
from functions import *
  

thres = 0.45 # Threshold to detect object
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

language='en'




classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
   



def empty(a):
    pass



cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",640,140)
cv2.createTrackbar("Threshold1","parameters",23,255,empty)
cv2.createTrackbar("Threshold2","parameters",20,255,empty)



#def return_filename():


def getC(img1,img2,objname,objd):#here img1 is taken as input and img2 as output but we are giving both of them at a time
    counter,_=cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    #draw the ocunter so we can get the outsideof the function
    #cv2.drawContours(img2,counter,-1,(178,34,34),7)

    for cout in counter:
        area=cv2.contourArea(cout)
        if area>2000:
            cv2.drawContours(img2,cout,-1,(178,34,34),2)
            # cv2.imshow("fdgdgf",img2)
            # cv2.waitKey(30000)
            peri=cv2.arcLength(cout,True)
            apx=cv2.approxPolyDP(cout,0.02*peri,True)
            x,y,width,height=cv2.boundingRect(apx)#y1,x1,y2,x2
            cv2.rectangle(img2,(x,y),(x+width,y+height),(0,255,0),2)

            #1.4cm=153
            #14mm=153
            #Ratio pixels to mm
            # rat=153/14
            # mm=width/rat
            # cm=round(mm/10,2)
            #print("cm : "+str(cm))
            area=int(area/37.79)
            width=int(width/37.79)
            height=int(height/37.79)




            classIds, confs, bbox = net.detect(img2,confThreshold=thres)
            #print(classIds,bbox)           
            if len(classIds) != 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    cv2.rectangle(img,box,color=(255,0,0),thickness=1)

                    cv2.putText(img2,classNames[classId-1].upper(),(box[0]+10,box[1]-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)

                    #print(classNames[classId-1].upper()+" "+str(round(confidence*100,2)))    

                    objname.append(str(classNames[classId-1].upper())+" "+str(round(confidence*100,2)))

                    cv2.putText(img2,str(round(confidence*100,2)),(box[0]+200,box[1]-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)


            cv2.putText(img2,"points:"+str(len(apx)),(x+width+20,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)
            cv2.putText(img2,"Area:"+str(int(area)),(x+width+20,y+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)
            cv2.putText(img2,"Height:"+str(height),(x+width+20,y+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)
            cv2.putText(img2,"Width:"+str(width),(x+width+20,y+80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)
            cv2.putText(img2,"Shape:"+str(shapedef(len(apx))),(x+width+20,y+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,128),2)

            print("["+str(len(apx))," ",(int(area))," ",width," ",height," ",shapedef(len(apx))+"]")
            s=[]
            s.append(str(len(apx)))
            s.append(str(int(area)))
            s.append(str(width))
            s.append(str(height))
            s.append(shapedef(len(apx)))
            objd.append(s)

        

cam_port = 0
cam = cv2.VideoCapture(cam_port)
res,img=cam.read()

if res:
    imgn="./upload/"
    #cv2.imshow(" ",img)
    cv2.imwrite(imgn+"cam_img.jpg",img)
    # cv2.waitKey(0)
    imgn=imgn+"cam_img.jpg"
    print(imgn+"cam_img.jpg")
    name="cam_img"

else:

    imgn="./upload/"
    imgn2=input("Enter file name: ")
    imgn+=imgn2
    name=names(imgn2)
    print(name)

a=int(0)

while True and a!=40:
    s1="the references are ,"
    s2=" "
    s3=""
    print(name," names of the ")
    img=cv2.imread(imgn)    #here onwards the img takes inp
    cv2.imshow("once",img)
    img2=img.copy()



        # 
        # wid=img.shape[1]
        # hei=img.shape[0]
        # print("width: "+str(wid)+"\nheight: "+str(hei)+"\n")
        # if (hei>600 and wid>600):
        #     img_half = cv2.resize(img, None, fx = 0.4, fy = 0.4)
        #     img=img_half
        #     cv2.imshow("changed or crossed img",img)                  
            
        #here onwards the img takes inp

    img2=img.copy()
    imgblur=cv2.GaussianBlur(img,(7,7,),1)#blur version but colored

            # imghsv=cv2.cvtColor(imgblur, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(src=imgblur, lowerb=np.array([0, 64, 153]), upperb=np.array([179, 255, 255]))
            # img_hsv_modify = cv2.bitwise_and(imgblur, imgblur, mask=mask)
        

    gray=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)#convert the colored img to blure img


            
    threshold1=cv2.getTrackbarPos("Threshold1","parameters")
    threshold2=cv2.getTrackbarPos("Threshold2","parameters")
            
            
    imgcanny=cv2.Canny(gray,threshold1,threshold2)
    kernel=np.ones((5,5))
    imgDil=cv2.dilate(imgcanny,kernel,iterations=3)
    objname,objd=[],[]

    getC(imgDil,img2,objname,objd)
            
    print(objname)
            # print(objd)

    for i in objname:
        s1+=i

    for i in objd:
        s=" , counters or points: "+i[0] +" , Area: "+i[1]+" , Width: "+i[2]+" cm , Height: " +i[3]+" cm , Shape: "+i[4]
        s2+=s
    s3=s1+s2
   
    print("points: "+objd[0][0])
    print("Area: "+objd[0][1])
    print("Width: "+objd[0][2])
    print("Height: ",objd[0][3])
    print("shape: "+objd[0][4])
        
    print(("*"*20)) 
            
    objname,objd=[],[]

    imgstack=stackImages(0.6,([img,gray,imgcanny],[img2,imgblur,imgDil]))#[img,gray,imgcanny],[img2,imgblur,imgDil]
    cv2.imshow("shape",imgstack)
    cv2.imshow("actual",img2)
    cv2.waitKey(1000)
    a+=1
    s4=""    
    s4=s3    
    s3+="\n\nIf you wanted to listen the output click 'yes'"
   

   
res=tkinter.messagebox.askquestion("Result", s3)
print(name)
if res=='yes':
    audio_coonvertion(s4,name,language,False)
    cv2.imwrite("./output/output_d"+name+".jpg",imgstack)
    cv2.imshow("k",imgstack)    
    cv2.imwrite("./output/output_"+name+".jpg",img2)
    cv2.imshow("output",img2)
    cv2.waitKey(0)
else:
    print("the data is not saved")   

    #cv2.waitKey(10000)
    # threshold1=cv2.getTrackbarPos("Threshold1","paramters")
    # threshold2=cv2.getTrackbarPos("Threshold2","paramters")
    # imgcanny=cv2.Canny(gray,threshold1,threshold2)
