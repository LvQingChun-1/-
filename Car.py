#coding:utf8
import math
import cv2
import numpy as np
def fillholes(gray):
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	res=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
	return res


def find_Contours(contours):
	error=0.4
	aspect=4.7272 			#宽高比
	min_area=38*aspect*38		#区域面积的最小像素 ,可调
#	min_area=15*aspect*15
	max_area=125*aspect*125		#区域面积的最大像素
	rmin_aspect=aspect-aspect*error
	rmax_aspect=aspect+aspect*error
	screenCnt=[]
	for c in contours:
		(x,y,w,h)=cv2.boundingRect(c)
		r=w/float(h)
		area=cv2.contourArea(c)
		if r<1:
			r=1/r
		if area>=min_area and area<=max_area and r>=rmin_aspect and r<=rmax_aspect:
			print cv2.contourArea(c)
			screenCnt.append(c)
	return screenCnt	

def rotate_center(scr,angle,scale=1):
	w=scr.shape[1]
	h=scr.shape[0]
	rangle=np.deg2rad(angle)  #将角度转化成弧度
	nw=(abs(np.sin(rangle)*h)+np.cos(rangle)*w)*scale
	nh=(abs(np.cos(rangle)*w)+np.sin(rangle)*h)*scale
	center=(nw*0.5,nh*0.5)
	rot_mat=cv2.getRotationMatrix2D(center,angle,scale)
	rot_move=np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))
	rot_mat[0,2]+=rot_move[0]
	rot_mat[1,2]+=rot_move[1]
	return cv2.warpAffine(scr,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))))		



img=cv2.imread('car.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x=cv2.Sobel(img,cv2.CV_16S,1,0)
y=cv2.Sobel(img,cv2.CV_16S,0,1)

absX=cv2.convertScaleAbs(x)
absY=cv2.convertScaleAbs(y)

dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
thresh=cv2.threshold(dst.copy(),100,255,cv2.THRESH_BINARY)[1]
thresh=fillholes(thresh)

#thresh=cv2.GaussianBlur(thresh,(11,11),0)
kernel=cv2.getStructuringElement(cv2.MORPH_DILATE,(7,7))
thresh=cv2.dilate(thresh,kernel)
thresh=cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
screenCnt=find_Contours(cnts)
print len(screenCnt)
mask=np.zeros(img.shape[:2],dtype=np.uint8)
for (j,m) in enumerate(screenCnt):
	cv2.drawContours(mask,[m],-1,(255,255,255),-1)
masked=cv2.bitwise_and(img,img,mask=mask)


for c in screenCnt:
	minRect=cv2.minAreaRect(c)
	box=cv2.boxPoints(minRect)
	box=np.int0(box)
	r=minRect[1][0]/minRect[1][1]
	angle=minRect[2]
	if r<1:
		angle+=90
	#将图片进行仿射变换
	print angle
	cv2.drawContours(masked,[box],0,(0,255,0),3)
	cv2.drawContours(img,[box],0,(0,255,0),3)


masked_1=rotate_center(masked,angle,1)
cv2.imshow('c',masked_1)
cv2.imshow('a',masked)
cv2.imshow('b',thresh)
cv2.imshow('a',img)
cv2.waitKey(0)
