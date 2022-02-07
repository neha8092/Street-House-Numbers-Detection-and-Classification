# Import necessary libraries
import cv2
import numpy as np
import os
import h5py
import random
#import matplotlib.pyplot as plt
from svhn.detection import Detection
#Create Train Data 

count=0
def process_images(train_path,image_names, labels,all_cropped_images, all_labels):
    #Loop over the images
    # Extract each image and label associated with that image
    # based on the length of the label, crop the image and reshape it
    # Save each image in the respective class folders
    
    for index, (image_name,lab) in enumerate(zip(image_names,labels)):
        try:
            bboxes= []
            print(index,image_name, lab)

            #check image extension
            if image_name.split('.')[-1] =='png':

                image= cv2.imread(train_path+image_name)
                #print("image", image.shape)
                #get each label
                labels= lab['label']
                for index, label in enumerate(labels):
                    #print("labels",labels)

                    y1,x1= lab['left'][index], lab['top'][index]
                    y2,x2= lab['left'][index]+lab['width'][index], lab['top'][index]+lab['height'][index]
                    #print("coords",y1,x1,y2,x2 )
                    #check for positive coords
                    if (x1 <0) or (x2<0) or (y1<0) or (y2 <0):
                        continue
                    else:
                        bboxes.append((x1,y1,x2,y2))
                        all_cropped_images, all_labels=crop_and_save(image, label, (x1,y1,x2,y2),all_cropped_images,all_labels)

                all_cropped_images, all_labels= get_non_text_regions(image,bboxes,all_cropped_images,all_labels)
                all_cropped_images, all_labels= mser_non_text(image,bboxes,all_cropped_images,all_labels)
            else:
                pass
        except:
            pass
    return all_cropped_images, all_labels
        #raise NotImplementedError

def mser_non_text(image,bboxes,all_cropped_images,all_labels):
    detection= Detection(image)
    mser_region, _= detection.mser()

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in mser_region]
    mser_boxes = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        mser_boxes.append((x, y, x + w, y + h))
    mser_boxes = np.array(mser_boxes)

    nms_boxes= detection.apply_nms(mser_boxes)

    # Compare with all the ground truth bboxes of that image
    # NMS boxes must not be in the range of ground truth boxes
    # get IOU between MSER and ground truth

    for boxB in nms_boxes:
        all_cropped_images, all_labels=get_iou(bboxes, boxB, all_cropped_images, all_labels, image)

    return all_cropped_images, all_labels


def get_non_text_regions(image,bboxes,all_cropped_images,all_labels):
    # Get the Non-text random region from the image and save in label 0.
    image = cv2.resize(image, (32, 32))
    all_cropped_images.append(image)
    label = 0
    all_labels.append(label)
    # Get the Non-text random region from the image and save in label 0.
    if image.shape[0]>32 and image.shape[1]>32:
        x1_rand= random.randint(1, image.shape[0]-32)
        y1_rand= random.randint(1,image.shape[1]-32)
        x2_rand, y2_rand= x1_rand+10, y1_rand+10
        #print(x1_rand,x2_rand, y1_rand,y2_rand)
        rand_img= image[x1_rand:x2_rand, y1_rand:y2_rand]
        rand_img= cv2.resize(rand_img, (32,32))
        #get the IOU and if IOU< Threshold then write it in Non-Text Class
        boxB= [x1_rand,y1_rand,x2_rand,y2_rand]
        #print("bboxes",len(bboxes))
        all_cropped_images, all_labels=get_iou(bboxes, boxB,all_cropped_images, all_labels,image)
    return all_cropped_images, all_labels

def get_iou(bboxes, boxB,all_cropped_images, all_labels,image):
    iou_vals = []
    for boxA in bboxes:
        iou_vals.append(iou(boxA, boxB))
        # print("iou", iou_vals)
    # print("iou_vals", iou_vals)
    if all(x < 0.15 for x in iou_vals) == True:
        label = 0
        #count = +1
        # print("Count",count)
        #print("image", image.shape, label, boxB)
        all_cropped_images, all_labels = crop_and_save(image, label, boxB,all_cropped_images,all_labels)
    else:

        return all_cropped_images, all_labels

    return all_cropped_images, all_labels

def crop_and_save(image, label, coords,all_cropped_images,all_labels):
    x1,y1,x2,y2= coords
    #print(x1,x2,y1,y2)
    cropped_label_img= image[x1:x2,y1:y2]
    #print("cropped_label_img",cropped_label_img.shape)
    if cropped_label_img.shape[0]>0 and cropped_label_img.shape[1]>0:
        #Reshape the image to 32x32 --can change this later
        cropped_label_img= cv2.resize(cropped_label_img, (32,32))
        all_cropped_images.append(cropped_label_img)
        all_labels.append(label)
    #print("all", len(all_cropped_images),(all_labels), all_cropped_images[0].shape)
    return all_cropped_images, all_labels
    

def get_image_name(f, idx,names):
    image_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(image_name)

def get_image_boxes(f, idx,bbox_attr,bboxes):
    meta = {key : [] for key in bbox_attr}
    box = f[bboxes[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return meta

def process_meta_file(file_path):
    file= h5py.File(file_path)
    #print(file['digitStruct'].keys())

    names = file['digitStruct/name']
    bboxes = file['digitStruct/bbox']

    image_names=[]
    bboxes_names= []
    bbox_attr = ['height', 'left', 'top', 'width', 'label']


    assert bboxes.shape[0]== names.shape[0]
    for i in range(0,names.shape[0]):  #for testing, change later
         
        #image names
        image_name = get_image_name(file,i, names)
        image_names.append(image_name)
        #image labels/bbox
        bbox= get_image_boxes(file, i,bbox_attr, bboxes)
        bboxes_names.append(bbox)
        #raise NotImplementedError

    return image_names, bboxes_names


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



def save_to_hdf5(samples, labels,filename):
    file=filename
    write_mode='w'
    db = h5py.File(file, write_mode)
    dataset = db.create_dataset("images", samples.shape, dtype='uint8')
    dataset[:] = samples[:]
    dataset = db.create_dataset("labels", labels.shape, dtype='int')
    dataset[:] = labels[:]
    db.close()


if __name__=='__main__':
    # CHANGE TO SUBMISSION PATH
    TRAIN_DATA_PATH= '../../data/train/train/'
    
    NUM_CLASSES=11
    
    file= TRAIN_DATA_PATH+'digitStruct.mat'
    image_names, labels= process_meta_file(file)
    #process_images(images)
    #print(len(image_names), len(labels))
    all_cropped_images = []
    all_labels = []
    # Process all the images and dump into respective label folders
    final_cropped_images, final_labels= process_images(TRAIN_DATA_PATH,image_names, labels,all_cropped_images,all_labels)

    samples= np.stack(final_cropped_images, axis=0)
    labels= np.array(final_labels).reshape(samples.shape[0],1)
    
    #Save samples and labels to hdf5 format

    filename='../../final_data/train_v7.hdf5'
    save_to_hdf5(samples,labels, filename)

    
    


    
