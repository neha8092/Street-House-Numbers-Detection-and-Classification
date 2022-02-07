import config
import cv2
import numpy as np



class Recognition:
    def __init__(self,image,bboxes,model) -> None:
        self.image= image
        self.bboxes= bboxes
        self.model= model
    

    def _post_processing(self):
        raise NotImplementedError


    def get_model_results(self):
        image= self.image.copy()
        for box in self.bboxes:
            x1, y1, x2, y2 = box

            cropped = self.image[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, (32, 32)).reshape(1, 32, 32, 3)
            val = self.model.predict(cropped)
            value = np.argmax(val)

            if value !=0:
                #draw a bbox across the image and the print digit
                image = cv2.rectangle(image, (x1, y1), (x2,y2), (0, 255, 0), 2)
                if value == 10:
                    value = 0
                cv2.putText(image, str(value), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)


        return image




    def recognize(self):
        image= self.get_model_results()
        return image



