import face_recognition
import os
import argparse
import json
from utils.inference import load_image
import numpy as np
import json
#import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--gallery_dir", default = "/Users/admin/Desktop/face-ml/TestGallery")
args, _ = parser.parse_known_args()

gallery = args.gallery_dir

def process_gallery(directory):
	
	encoded_faces = []
	file_paths = []
	
	# encodes face in each image of gallery
	for filename in os.listdir(directory):
		# checks for non_image files
		if filename.endswith(".jpg") or filename.endswith(".png"):
			try:
				#image = load_image(directory+'/'+filename)
				#image = image.astype('uint8')
				image = face_recognition.load_image_file(directory+'/'+filename)
				encoding = face_recognition.face_encodings(image)
				encoded_faces.append(encoding)
				file_paths.append(filename)
				#print(filename, "processed")
			except:
				print("UNABLE TO PROCESS" , filename)
				continue
		else:
			print("UNABLE TO PROCESS", filename)

	return zip(file_paths, encoded_faces)

#zip(file_paths, encoded_faces)


# a, b = process_gallery(gallery_path)
# I wanted to store the processed information in system memory, so that we would only
# have to run this program once
'''
with open('ProcessedGallery.txt', 'wb') as fp:
	pickle.dump(file_paths, fp)
	
string = path + "\n" + face_in_str +"\n\n"
print(type(string))
json.dump(string, thefile)
'''

file_paths, encoded_faces = zip(*process_gallery(gallery))

thefile = open('ProcessedGallery.txt', 'w')
json.dump(file_paths, thefile)
faces_in_str = []
for face in encoded_faces:
	face = np.squeeze(face)
	face_in_str = (''.join(map(str,face)))
	faces_in_str.append(face_in_str)
json.dump(faces_in_str, thefile)
thefile.close()

'''
for path, face in process_gallery(gallery):
	face = np.squeeze(face)
	face_in_str = str(face)
	#face_in_str = (' '.join(map(str,face)))
	thefile.write(path + "\n" + face_in_str +"\n\n")
'''
