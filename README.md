# RASPBERRYPI-FACE_RECOGNITION

## DESCRIPTION:

This is a raspberry pi python project for face recognition and speaks the name of person using gTTS(Gooogle text to speak),using 2x15W speakers.

## COMPONENTS:

1.Raspberry pi 4 or 3,(Raspberry pi 4 was used in this project)
2.Camera module
3.Regulated power supply

## Circuit Diagram:

![raspberry pi camera diagram](https://user-images.githubusercontent.com/102857010/184334097-6cc1b7af-0c73-4431-9906-3fdd50679608.jpeg)

## Working:

First we have to setup our camera for raspberry pi,follow the following guide to setup your camera
https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/0
After setting up the camera we have to install following libraries on our raspberry pi:
`````
sudo pip install opencv-contrib-python
sudo nano /etc/dphys-swapfile
Once the file is open, comment out the line CONF_SWAPSIZE=100 and add CONF_SWAPSIZE=2048.
Press Ctrl-X, Y and then Enter to save your changes to dphys-swapfile.
sudo systemctl restart dphys-swapfile
sudo pip install face_recognition
sudo pip install imutils.video
sudo pip install pickle

`````
First we have to create a Dataset of your photos: (Change the 'Ayush' from the code to your name), this code will create a folder name "dataset" which will have another folder with your name, which will contain all your images.After running this code click 'space' from your keyboard to take pictures, and once you are done press 'ESC' from your keyboard.

## CODE:
``````
import cv2

name = 'Ayush' #replace with your name

cam = cv2.VideoCapture(0)

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("press space to take a photo", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
``````
Then after we create a dataset folder now we train our images using a python code:

## CODE:
``````
#! /usr/bin/python

# import the necessary packages
from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="hog")

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
``````

## CODE:

After training your images its time to run your code:

`````
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import os
from playsound import playsound
import sys
import os
from playsound import playsound
from gtts import gTTS
import time
def speak():
  tts = gTTS('Welcome back sir', lang='en', tld='ca',slow=False)
  tts.save('hello.mp3')
  time.sleep(0.5)
  playsound("hello.mp3")
  time.sleep(10)
  os.remove("hello.mp3")
def speakUNKNOWN():
  tts = gTTS('Unidentified personal,You dont have access to this computer', lang='en', tld='ca',slow=False)
  tts.save('hello.mp3')
  playsound("hello.mp3")
  time.sleep(10)
  os.remove("hello.mp3")
# Fetch the service account key JSON file contents
# Initialize the app with a service account, granting admin privileges
#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
#vs = VideoStream(src=2,framerate=10).start()
vs = VideoStream(0).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
  # grab the frame from the threaded video stream and resize it
  # to 500px (to speedup processing)
  frame = vs.read()
  frame = imutils.resize(frame, width=500)
  # Detect the fce boxes
  boxes = face_recognition.face_locations(frame)
  # compute the facial embeddings for each face bounding box
  encodings = face_recognition.face_encodings(frame, boxes)
  names = []

  # loop over the facial embeddings
  for encoding in encodings:
      # attempt to match each face in the input image to our known
      # encodings
      matches = face_recognition.compare_faces(data["encodings"],
          encoding)
      name = "Unknown" #if face is not recognized, then print Unknown

      # check to see if we have found a match
      if True in matches:
          # find the indexes of all matched faces then initialize a
          # dictionary to count the total number of times each face
          # was matched
          matchedIdxs = [i for (i, b) in enumerate(matches) if b]
          counts = {}

          # loop over the matched indexes and maintain a count for
          # each recognized face face
          for i in matchedIdxs:
              name = data["names"][i]
              counts[name] = counts.get(name, 0) + 1

          # determine the recognized face with the largest number
          # of votes (note: in the event of an unlikely tie Python
          # will select first entry in the dictionary)
          name = max(counts, key=counts.get)

          #If someone in your dataset is identified, print their name on the screen
          if currentname != name:
              currentname = name



      # update the list of names
      names.append(name)

  # loop over the recognized faces
  for ((top, right, bottom, left), name) in zip(boxes, names):
      # draw the predicted face name on the image - color is in BGR
      cv2.rectangle(frame, (left, top), (right, bottom),
          (0, 255, 225), 2)
      y = top - 15 if top - 15 > 15 else top + 15
      cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
          .8, (0, 255, 255), 2)
      if name=="Unknown":
          speakUNKNOWN()
          sys.exit()
      if name!="Unknown":
          speak()
          sys.exit()
  # display the image to our screen
  cv2.imshow("Facial Recognition is Running", frame)
  key = cv2.waitKey(1) & 0xFF
  # quit when 'q' key is pressed
  if key == ord("q"):
      break

  # update the FPS counter
  fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
``````

