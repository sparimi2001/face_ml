import cv2
import face_recognition
import os
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def output_text(men, women):
    if men == 1:
        print("1 man found")
    elif men > 1:
        print("%s men found" %(men))
    
    if women == 1:
        print("1 woman found")
    elif women > 1:
        print("%s women found" %(women))

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

# returns CascadeClassifier object based on data from model_path (used for detection)
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

# uses CascadeClassifier object to detect faces
def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

# draws box on image using location of face; thickness of 2
def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors