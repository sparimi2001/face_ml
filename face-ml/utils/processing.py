import cv2
import face_recognition
import os
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def check_percent(pil_image, pil_face, percent):
    image_size = pil_image.size[0] * pil_image.size[1]
    face_size = pil_face.size[0] * pil_face.size[1]
    if (face_size/image_size) >= (percent/100):
        return True
    else:
        return False

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

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
                encoding = face_recognition.face_encodings(image)[0]
                encoded_faces.append(encoding)
                file_paths.append(filename)
                #print(filename, "processed")
            except:
                print("UNABLE TO PROCESS" , filename, "in gallery")
                continue
        else:
            print("UNABLE TO PROCESS", filename, "in gallery")

    return zip(file_paths, encoded_faces)