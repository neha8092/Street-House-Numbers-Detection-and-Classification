import streamlit as st
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
from PIL import Image
import numpy as np

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
    
   
    DEMO_IMAGE= './input/3.png'

    st.title("     House Numbers- Detection and Classification      ")
    
    demo_image = DEMO_IMAGE
    demo_image= cv2.imread(DEMO_IMAGE)
    demo_image=cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
    final_image= run(demo_image)

    st.image(
            final_image, caption=f"Sample Processed image", width=150,
                 )
    
    st.header("Above is the sample Output.")
    st.header("Upload an image from with a House Number.")
    
    st.write("Note: You can also test on the sample images in my github repo in the input/ folder")
    
    file_uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    class_btn = st.button("               Detect and Classify                ")
    
    if file_uploaded is not None:    
        #image = Image.open(file_uploaded)
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image=cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.image(opencv_image, caption='Uploaded Image', width=150)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                final_image = run(opencv_image)
                st.image(
                    final_image, caption=f"Processed image", width=150,
                      )
    
    st.write("Note: There could be some false positives when detecting the house numbers. I will be improving the same with time.. Till then, please use the existing model :) ")
    

    


