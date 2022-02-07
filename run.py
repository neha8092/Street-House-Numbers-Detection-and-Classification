
import cv2
import os
from detection import Detection
from recognition import Recognition
import config
import h5py
from tensorflow.keras.models import load_model
import tensorflow as tf
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

def run(image):
    #detection->recognition->save the image
    detection = Detection(image)
    final_detected_boxes= detection.detect()
    if final_detected_boxes is None:
        return image
    #print("final_detected_boxes",len(final_detected_boxes))

    # Recognition
    #Load the model
    model = load_model(config.MODEL_PATH)

    recognition= Recognition(image, final_detected_boxes, model)
    final_image= recognition.recognize()
    return final_image
    


if __name__=='__main__':

    PATH= './input/'
    OUT_PATH='./'
    #Load Images
    images= os.listdir(PATH)

    for i,image in enumerate(images):

        #print(image)
        img= cv2.imread(PATH+image)

        # Loop over the images and run algo
        final_image = run(img)

        cv2.imwrite(OUT_PATH+image, final_image)



