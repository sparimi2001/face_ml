import face_recognition

import sys
import os
import argparse
import cv2

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

from utils.processing import load_image
from utils.processing import apply_offsets
from utils.processing import process_gallery
from utils.processing import check_percent

#class ImageError(Exception):
	#exit()

# parameters for loading data and images
parser = argparse.ArgumentParser(description = "classify gender and identify face in image")
parser.add_argument('-i', '--image_dir', type = str, \
	default = "TestImages", \
	help = 'give the path to the directory of images you want to analyze')
parser.add_argument('-g', '--gallery_dir', default = 'Gallery', \
	help = 'give the path to the gallery of known faces')
parser.add_argument('-m', '--min-face-area-percent', type = int, default = 0,\
	help = 'give a percentage between 0 and 100 to check if the face takes up at least a certain percentage of the entire image')
parser.add_argument('-o', '--offset', type = int, default = 10, \
	help = 'give a value for detection offset (default = 10)')

args, _ = parser.parse_known_args()

gallery = args.gallery_dir
known_file_paths, known_face_encodings = zip(*process_gallery(gallery))

image_dir = args.image_dir

# path to saved CascadeClassifier model
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
# path to saved keras model
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
# based on gender_classifier keras.models.load_model.predict(), which uses trained model
gender_labels = {0:'woman',1:'man'}

min_percent = args.min_face_area_percent

# Adds buffer to face coordinates (optional arument)
offset = args.offset
gender_offsets = (offset,offset)

# loading models: 
	# face_detection: loads CascadeClassifier object from file
	# gender_classifier: loads Keras object from file
face_detection = cv2.CascadeClassifier(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# finds size of image needed to enter into classifier: tuple of ints (img_height, img_width)
gender_target_size = gender_classifier.input_shape[1:3]

#loops through each image in test directory
for filename in os.listdir(image_dir):

	print("\n" + filename, "is being analyzed")
	image_path = image_dir + '/'+ filename

	#loading images: converts image into an array using keras: gray needed to detect face; rgb needed to classify gender; f_image needed to encode face
	if filename.endswith(".jpg") or filename.endswith(".png"):
		try:
			gray_array = load_image(image_path, grayscale=True)
			rgb_pil_image = image.load_img(image_path, grayscale = False, target_size = None)
			rgb_array = image.img_to_array(rgb_pil_image)
			#face_recognition and gender_classification both use arrays to analyze images; face_recognition just uses a different type
			f_image_array = rgb_array.astype('uint8')
		except:
			print("UNABLE TO LOAD", image_path)
			continue
	else:
		print(filename, "is not jpg or png")
		continue

	# removes single-dimensional entries from shape of an array and converts data from float32 into uint8 (most likely to reduce space?)
	# gray has one less dimension: only needs one number to represent color, squeezing is uneccessary I believe, but makes it simpler
	gray_array = np.squeeze(gray_array)
	gray_array = gray_array.astype('uint8') #??

	# uses CascadeClassifier method, detectMutliScale, to detect faces; detectMutliScale(img, scaleFactor, minNeighbors)
	# returns list of rectangles: coordinate format: x, y, width, height
	faces = face_detection.detectMultiScale(gray_array, 1.3, 5)
	# stores features of each face in a 2D array
	encodings = face_recognition.face_encodings(f_image_array)

	if len(faces) > 1 or len(encodings) > 1:
		print("Error: More than one face found")
		continue
		#raise ImageError
	elif len(faces) == 0 or len(encodings) == 0:
		print("Error: No faces found")
		continue
		#raise ImageError

	# takes first value for each array (there should only be one face)
	face_coordinates = faces[0]
	face_encoding = encodings[0]

	# compares the unknown face to the gallery and prints the file with a matching face
	matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
	matching_file_path = "Match not found"
	if True in matches:
		first_match_index = matches.index(True)
		matching_file_path = known_file_paths[first_match_index]
		print("Match from gallery:", matching_file_path)
	else:
		print(matching_file_path)

	# zooms in on faces. Has some extra space (offsets) to increase accuracy of gender
	# sometimes, the offset goes past the boundaries of the image if the face is too close to the edge
	try:
		x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
		face_array = rgb_array[y1:y2, x1:x2]
		print(x1, x2, y1, y2)
	except:
		print("\n", x1, x2, y1, y2)
		print("If any of these values are negative, try a smaller offset, like 0. The face may be too close to the edge.\n")
		pass
		
	# checks how large the face is relative to the entire image
	if min_percent > 0:
		face_image = image.array_to_img(face_array)
		large_enough = check_percent(rgb_pil_image,face_image,min_percent)
		if large_enough:
			pass
		else:
			print("Warning: the face takes up less than %s%% of the image" %(min_percent))

	# resizes image to fit target (based on trained models)	
	face_array = cv2.resize(face_array, (gender_target_size))

	# prediction software expects 4 dimensional array
	face_array = face_array / 255.0
	face_array = np.expand_dims(face_array, 0)

	# gives list: likelihood of being man and likelihood of being woman (sum to 1)
	gender_prediction = gender_classifier.predict(face_array)

	# returns index of larger likelihood, uses index to find gender, prints the gender
	gender_label_arg = np.argmax(gender_prediction)
	gender_text = gender_labels[gender_label_arg]
	print("Gender:", gender_text)