print("Importing dependecies...")

from utils.inference_classification import Classify
from utils.inference_object_detection import Detect
from utils.preprocess import transform_image, add_zero_pixels_atas
import os
from utils.serial_port import Micro
import cv2 as cv
import time
from datetime import date, datetime
import threading
import queue
from utils.run_models import Models
import numpy as np
from datetime import datetime
from copy import deepcopy
import time

ROOT = os.getcwd()
print("Preparing model...")



models = Models(
    model_path= (
        
            os.path.join(
            ROOT,
            "models",
            "bottle",
            "bottle_best_ver7.xml",
            ),
            # os.path.join(
            # ROOT,
            # "models",
            # "condition",
            # "best.xml",
            # ),
            # os.path.join(
            # ROOT,
            # "models",
            # "brand",
            # "brand_best_ver5.xml"
            # ),
        

    ),
    class_name=(
        ["bottle", "not_bottle"],
        # ["deformed", "not_deformed"],
        # ["aqua", "coca_cola", "le_minerale", "unknown"]
    )

)

# model_check_if_bottle = Classify(
#     # dont forget to change this
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_botol8novb1b2",
#         "mobilenet_botol8novb1b2.xml",
#     ),
#     class_name=["bottle", "not_bottle"],
# )

# model_check_condition = Classify(
#     # dont forget to change this
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_botol8novb1b2",
#         "mobilenet_botol8novb1b2.xml",
#     ),
#     class_name=["deformed", "good"],
# )

# model_check_size = Classify(
#     # dont forget to change this
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_botol8novb1b2",
#         "mobilenet_botol8novb1b2.xml",
#     ),
#     class_name=["large",'medium','small'],
# )

# model_check_brand = Classify(
#     # dont forget to change this
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_botol8novb1b2",
#         "mobilenet_botol8novb1b2.xml",
#     ),
#     class_name=['aqua','cocacola','leminerale','other'],
# )


# model_botol = Classify(
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_botol8novb1b2",
#         "mobilenet_botol8novb1b2.xml",
#     ),
#     class_name=["botol", "not_botol"],
# )

# model_label = Classify(
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_label8novb1b2",
#         "mobilenet_label8novb1b2.xml",
#     ),
#     class_name=["label_peeled", "labelled"],
# )

# model_deformed = Classify(
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_deformedb1b2",
#         "mobilenet_deformedb1b2.xml",
#     ),
#     class_name=["deformed", "not_deformed"],
# )

# model_clean = Classify(
#     model_path=os.path.join(
#         ROOT,
#         "models",
#         "mobilenet_cleanb1b2",
#         "mobilenet_cleanb1b2.xml",
#     ),
#     class_name=["clean", "not_clean"],
# )

# object_detection = Detect(model_path=os.path.join(ROOT, "openvino_model", "best.xml"))

def debug_is_object_inside(command = 'no object'):
    if command == 'object inside':
        return True
    elif command == 'no object':
        return False

# def is_object_inside():
#     pass

# def predict_object_attribute(transformed_image):

#     pass
    # if 
    


# def deform_check(transformed_image, orig_image):
#     is_deformed = model_deformed.predict(transformed_image)
#     size = object_detection.predict(orig_image)[0]

#     if is_deformed == "deformed":
#         is_labelled = model_label.predict(transformed_image)
#         return [is_deformed, is_labelled]

#     is_labelled = model_label.predict(transformed_image)

#     if is_labelled == "labelled":
#         return [is_deformed, size, is_labelled]

#     is_clean = model_clean.predict(transformed_image)
#     return [is_deformed, size, is_labelled, is_clean]


class VideoCapture:
    def __init__(self, name):
        self.cap = cv.VideoCapture(name)
        self.q = queue.Queue()
        # self.keyboard_input =''
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        
        t1 = threading.Thread(target=self._reader)
        t1.daemon = True

        # t2 = threading.Thread(target=self.read_keyboard)
        # t2.daemon = True

        t1.start()
        # t2.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            # cv.imshow('Real time camera', frame)
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        print(f'\nReleasing camera...')
        self.cap.release()
        # capture.release()
        print(f'\nCamera has been released successfully.')
        print(f'\nClosing all windows...')
        # cv.destroyAllWindows()
        cv.destroyAllWindows()
        print(f'\nAll windows have been closed successfully.\n')
        

    # def read_keyboard(self):
    #     self.keyboard_input = cv.waitKey(10)
        

    def read_frame(self):
        return self.q.get()
    
    # def read_keyboard(self):
    #     while True:
    #         self.keyboard_input = cv.waitKey(1)

def preserved_aspect_ratio_resize(img: np.ndarray, 
                                  target_width = 640) -> np.ndarray:
    """
    Resizing the image to desired width while preserving the aspect ratio
    
    :param img: Image
    :type img: np.ndarray
    :param target_width: desired width
    :return: resized image with its aspect ratio preserved
    :rtype: ndarray[Any, Any]
    """
    aspect_ratio = img.shape[0]/img.shape[1]

    # if aspect_ratio > 1:
    transformed_img = cv.resize(img, 
                                (target_width, int(target_width*aspect_ratio)), 
                                interpolation=cv.INTER_LINEAR)
    # else:
    #     transformed_img = cv.resize(img, 
    #                                 (target_width, target_width/aspect_ratio), 
    #                                 interpolation=cv.INTER_LINEAR)        

    return transformed_img

def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Creating a square image by padding top-bottom or left-right depending which one between width and height is greater.
    Whichever smaller will be pad so that it become of the same length of the greater.
    If image is already square, will return the original image.
    
    :param img: Image
    :type img: np.ndarray
    :return: Padded to square image
    :rtype: ndarray[Any, Any]
    """
    if img.shape[0] == img.shape[1]:
        return img

    if img.shape[0] > img.shape[1]: # if width is greater than height
        pad_length = int((img.shape[0] - img.shape[1])/2)
        transformed_img = cv.copyMakeBorder(img, 
                                            0, 0, 
                                            pad_length, pad_length,
                                            cv.BORDER_CONSTANT, value=(114,114,114))
    else:
        pad_length = int((img.shape[1] - img.shape[0])/2)
        transformed_img = cv.copyMakeBorder(img, 
                                            pad_length, pad_length, 
                                            0, 0, 
                                            cv.BORDER_CONSTANT, value=(114,114,114))

    return transformed_img
    

if __name__ == "__main__":
    model_img_size = 640
    model_index = 0
    micro = Micro()
    capture = VideoCapture(1)
    micro.debug_send_command("start")
    counter = 0

    result = []
    annotated_frame = []
    for i in range(len(models.model_path)):
        result.append("")
        annotated_frame.append("")

    while True:
        counter += 1
        # _input = input("Enter index: ")
        # time.sleep(1)
        # print("There is a bottle!")
        
        frame = capture.read_frame()
        # try:
        #     frame = cv.imread(f'test_image_{_input}.jpg')
            
        # # cv.waitKey(0)
        # # frame = transform_image(frame)
        # # print(frame.shape)
            
        # except Exception:
        #     print(f'invalid index')
        #     raise(Exception)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = preserved_aspect_ratio_resize(frame, 640)
        frame = pad_to_square(frame)
        # cv.imshow('test', frame)
        # cv.imwrite("result.jpg", frame)
        # cv.waitKey(0)
        
        check = 0

        for i in range(len(models.model_path)):
            check += 1
            print(f"checkpoint {check}")
            
            result[i] = models.predict(frame, i)
            print(result[i])

            annotated_frame[i] = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
            # print(result)
            if result[i].numel() != 0:
                result[i] = result[i][0]
                check += 1
                print(f"checkpoint {check}")
                print(f'xyxy: {result[:4]}')
                print(f'class: {models.class_name[i][int(result[i][5])]}')
                print(f'confidence: {result[i][4]}')

                center_x = (result[i][0] + result[i][2]) / 2
                center_y = (result[i][1] + result[i][3]) / 2
                width = result[i][2] - result[i][0]
                height = result[i][3] - result[i][1]


                # if frame.shape[0] != frame.shape[1]:
                #     aspect_ratio = frame.shape[0]/frame[1]
                #     print(aspect_ratio)

                # else:
                #     xyxy_new = result[i][:4] 
                #     xyxy_new *= frame.shape[0] / model_img_size

                #     annotated_frame[i] = cv.rectangle(frame, 
                #                                    (int(xyxy_new[0]), int(xyxy_new[1])), 
                #                                    (int(xyxy_new[2]), int(xyxy_new[3])), 
                #                                    (255,0,0), 
                #                                    5)
                #     annotated_frame[i] = cv.putText(annotated_frame, 
                #                                  f'{models.class_name[i][int(result[i][5])]}', 
                #                                  (int(xyxy_new[0]), int(xyxy_new[1]) + 5), 
                #                                  cv.FONT_HERSHEY_SIMPLEX, 
                #                                  1, 
                #                                  (0, 255, 0), 
                #                                  2)
                #     # cv.imshow("test",annotated_frame)
                #     # cv.waitKey(0)

            
                xyxy_new = result[:4] 

                if model_img_size != frame.shape[0] and model_img_size != frame.shape[1]:
                    xyxy_new *= frame.shape[0] / model_img_size

                annotated_frame[i] = cv.rectangle(
                                            annotated_frame[i], 
                                            (int(result[i][0]), int(result[i][1])), 
                                            (int(result[i][2]), int(result[i][3])), 
                                            (0,0,255), 
                                            2
                                    )

                check += 1
                print(f"checkpoint {check}")
                print(len(annotated_frame))
                annotated_frame[i] = cv.putText(
                                            annotated_frame[i], 
                                            f'{models.class_name[i][int(result[i][5])]}: {int(result[i][4]*100)} %', 
                                            (int(result[i][0]), int(result[i][1]) - 10), 
                                            cv.FONT_HERSHEY_SIMPLEX, 
                                            0.6, 
                                            (0, 0, 255), 
                                            2
                                    )
                print(i)
            # cv.imwrite(f'capture_{datetime.strftime('%Y_%m_%d_%H_%M_%S_%f')}',frame)
            # frame = cv.putText(frame, 
            #                     f'{counter}', 
            #                     (10, 40), 
            #                     cv.FONT_HERSHEY_SIMPLEX, 
            #                     0.9, 
            #                     (0, 255, 0), 
            #                     2)
            
                # cv.imshow("test",annotated_frame)
                # cv.waitKey(0)
            # for i in range(len(models.model_path)):
        for i in range(len(models.model_path)):
            cv.imshow(f"model {i}", annotated_frame[i])

        if cv.waitKey(1) == ord('q'):
            break       

                
            

        
    
        