from keras.models import load_model

from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
    model = load_model('model/keras/facenet_keras.h5')
	
model.summary()
################################################################################################
import os
import cv2
import glob
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#################################################################################################
def distance(embeddings1, embeddings2):
	diff = np.subtract(embeddings1, embeddings2)
	dist = np.sum(np.square(diff),1)
	return dist

def most_similar(y, y_true , names):
	min_d = distance(y, y_true[0])
	min_i = 0
	for i in range(1,len(y_true)):
		print(i)
		d = distance(y,y_true[i])
		if(d < min_d):
			min_d = d
			min_i = i
	return names[min_i]


#################################################################################
y_true = []
paths = (glob.glob(r'images/single_people/**/*.jpg'))
print(paths)
names = []
for path in paths:
	# faces = None
	img = cv2.imread(path)
	print(img.shape)
	img = cv2.resize(img, (640,640))
	names.append(path.split('\\')[1])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# print(faces)
	if len(faces) == 0:
		continue
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	input_ = cv2.resize(roi_color, (160,160))
	print('Input_shape', input_.shape)
	y = model.predict_on_batch(np.array([input_])/255.0)
	# print('Y_true', y)
	y_true.append(y)
	cv2.imshow('I', img )
	cv2.waitKey(0)
print(len(y_true))
cv2.destroyAllWindows()



###########################################################################################

cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.rectangle(img, (x-20,y-20), (x+w+20, y+h+20),(0,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = rgb[y-20:y+h+20, x-20:x+w+20]
		
	print('Actua shape', roi_color.shape)
	input_ = cv2.resize(roi_color, (160,160))
	print('Input_shape', input_.shape)
	y = model.predict_on_batch(np.array([input_])/255.0)
	# print('Y_true', y)
	name = most_similar(y,y_true, names)
	cv2.putText(img, name,org= (100,100), fontFace =cv2.FONT_HERSHEY_SIMPLEX ,
				fontScale = 1 ,color =(255,0,0), thickness =3 )
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
