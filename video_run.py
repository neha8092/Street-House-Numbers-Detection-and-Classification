import cv2
import os
from svhn.detection import Detection
from svhn.recognition import Recognition
import config
import h5py
from tensorflow.keras.models import load_model
import tensorflow as tf


configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


def run(image):
    # detection->recognition->save the image
    detection = Detection(image)
    final_detected_boxes = detection.detect()
    #print("final_detected_boxes", len(final_detected_boxes))
    if final_detected_boxes is None:
        return image
    # Recognition
    # Load the model
    model = load_model(config.MODEL_PATH)

    recognition = Recognition(image, final_detected_boxes, model)
    final_image = recognition.recognize()
    return final_image


if __name__ == '__main__':

    video_path='../../final_data/tests/videos/test5.mp4'
    VIDEO_OUT_PATH='../../final_data/test_results_v2/test_vid_v5_out.avi'

    vid = cv2.VideoCapture(video_path)
    print("vid", vid)
    writer = None
    (W, H) = (None, None)
    count=0
    # loop over frames from the video file stream
    while True:
        count=count+1
        print(count)
        (res, frame) = vid.read()
        print(res)
        if not res:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        final_frame= run(frame)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, 10,
                                     (W, H), True)
        #cv2.imshow('frame',final_frame)
        # write the output frame to disk
        writer.write(final_frame)




