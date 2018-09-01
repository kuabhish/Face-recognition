# Face-recognition

This project requires some dependencies like:
keras, tensorflow, opencv
and most important weights which can be downloaded from 
https://drive.google.com/open?id=14e6A3O_ylCympEIRKl7WvQLQyLvyXNx4

Keep this file inside a directory named model/

The data file structure:
Please collect your dataset by collecting an image of yours and some other person just for going through this code.
images/ folder contains two directory :
single_people/ and two_people(two people for demo purpose):
single_people/ contains the directory with the name of the person and inside the directory it contains the images
(ex. /images/single_people_Abhishek/my.jpg)
two_people/ contains a folder with image of two people. 

test2.py descriptoion:
This file uses a set of images to compare and display the name of the person in front of the webcam

test3.py description:
This file uses the image in the two_people folder to identify each person from the names in single_people folder. 

