import config
import cv2
import numpy as np
import h5py
# from  skimage import exposure

class Detection:
    def __init__(self, image) -> None:
        self.image= image

        self.latest_regions=[]
        self.latest_bboxes=[]
        self.n1, self.n2, self.n3, self.n4, self.n5, self.n6 = [0] * 6
    
    
    def filter_regions(self, regions, bboxes):
        #Get the Image edges
        canny_img= cv2.Canny(image=self.image, threshold1=100, threshold2=200)
        #total number of MSER regions
        self.n1 = len(regions)
        #print("total number of MSER regions",self.n1)

        final_regions= []

        for i, region in enumerate(regions):
            final_regions= self.apply_geometric_props(canny_img, region, final_regions)
        
        #self.latest_regions= final_regions
        #print("total number of MSER regions After Geometric Properties",self.n6, len(final_regions))

        if len(final_regions)==0:
            return None
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in final_regions]
        final_bboxes = []
        for c in hulls:
            x, y, w, h = cv2.boundingRect(c)
            final_bboxes.append((x, y, x + w, y + h))
            #cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        final_bboxes= np.array(final_bboxes)
        #self.latest_bboxes= final_bboxes
        final_nms_boxes= self.apply_nms(final_bboxes)
        return final_nms_boxes
    
    def _getArea(self, region):
        area= len(list(region)) #Number of pixels in that region
        return area
    
    def _getPerimeter(self,image,region):
        # first get the bbox around that region and then get the number of pixels where value in non-zero
        # pass the edge detected image for good results
        x, y, w, h = cv2.boundingRect(region)
        perimeter= len(np.where(image[y:y+h, x:x+w] != 0)[0])
        return perimeter

    def _getAspectRatio(self,region):
        shape= max(region[:, 1]) - min(region[:, 1]), max(region[:, 0]) - min(region[:, 0])
        aspect_ratio= 1.0 * max(shape) / min(shape) + 1e-4
        return aspect_ratio

    def _getOccupyRate(self,region):
        shape= max(region[:, 1]) - min(region[:, 1]), max(region[:, 0]) - min(region[:, 0])
        occupancy_rate = (1.0 * self._getArea(region)) / (shape[0] *  shape[1] + 1.0e-10)
        return occupancy_rate

    def _getCompactness(self, image, region):
        compactness= (1.0 * self._getArea(region)) / (1.0 * self._getPerimeter(image,region) ** 2)
        return compactness

    def apply_geometric_props(self, canny_img, region, final_regions):

        if self._getArea(region) > self.image.shape[0] * self.image.shape[1] * config.AREA_LIM:
            #regions meeting area criteria
            self.n2 += 1
            if self._getPerimeter(canny_img,region) > 2 * (self.image.shape[0] + self.image.shape[1]) * config.PERIMETER_LIM:
                #regions meeting perimeter criteria
                self.n3 += 1
                if self._getAspectRatio(region) < config.ASPECT_RATIO_LIM:
                    #regions meeting aspect ratio criteria 
                    self.n4 += 1
                    if (self._getOccupyRate(region) > config.OCCUPATION_LIM[0]) and  (self._getOccupyRate(region) < config.OCCUPATION_LIM[1]):
                        self.n5 += 1
                        if (self._getCompactness(canny_img,region) > config.COMPACTNESS_LIM[0]) and (self._getCompactness(canny_img,region) < config.COMPACTNESS_LIM[1]):
                            self.n6 += 1

                            final_regions.append(region) #Final regions meeting all the geometric properties
        return final_regions
        

    def apply_nms(self, final_bboxes):

        nms_box_idx = []
        x1,y1, x2, y2= final_bboxes[:, 0], final_bboxes[:, 1], final_bboxes[0:, 2], final_bboxes[:, 3]
        # Get area
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sorting based on bottom right corner
        idx_sorted = np.argsort(y2)

        # delete duplicate boxes
        while len(idx_sorted) > 0:
            last = len(idx_sorted) - 1
            i = idx_sorted[last]
            nms_box_idx.append(i)

            # Find the maximum and minimum coordinates in the remaining frames
            xx1 = np.maximum(x1[i], x1[idx_sorted[:last]])
            yy1 = np.maximum(y1[i], y1[idx_sorted[:last]])
            xx2 = np.minimum(x2[i], x2[idx_sorted[:last]])
            yy2 = np.minimum(y2[i], y2[idx_sorted[:last]])

            # Calculate the  IoU
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            iou = (w * h) / area[idx_sorted[:last]]

            # If IoU is greater than the specified threshold, delete
            del_arr= np.concatenate(([last], np.where(iou > config.NMS_OVERLAP_THRESHOLD)[0]))
            idx_sorted = np.delete(idx_sorted, del_arr)

        #get the final NMS boxes
        final_nms_boxes= final_bboxes[nms_box_idx].astype("int")

        return final_nms_boxes
            

    def mser(self):

        # Grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Removing Noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 4, 9, 21)
        #contrast = exposure.equalize_adapthist(denoised, clip_limit=0.03)
        #contrast = (contrast * 255).astype('uint8')
        # Apply MSER and get all the blob regions
        mser = cv2.MSER_create()
        regions,bboxes = mser.detectRegions(denoised)
        self.latest_regions= regions
        self.latest_bboxes= bboxes
        return regions, bboxes

    
    
    def detect(self):
        regions, bboxes= self.mser()
        final_nms_boxes= self.filter_regions(regions, bboxes)
        return final_nms_boxes