from typing import List
from openvino.runtime import Core
import openvino
from .detector_utils import non_max_suppression, save_output
from .preprocess import preprocess_image_yolov8_format
import torch
import numpy as np

class Models:
    def __init__(self, model_path, class_name) -> None:
        if isinstance(model_path, str):
            self.model_path = (model_path)
        elif isinstance(model_path, list):
            self.model_path = tuple(model_path)
        else:
            self.model_path = model_path

        if len(self.model_path) != len(class_name):
            raise Exception(f'{len(self.model_path)} model(s) was given but class name for {len(class_name)} model(s) was given')
        
        self.core = Core()
        self.models = []
        self.compiled_models = []
        self.input_layer_ir = []
        self.output_layer_ir = []
        # print(self.model_path)
        for model in self.model_path:
            
            self.models.append(self.core.read_model(model=model))
        for model in self.models:
            self.compiled_models.append(self.core.compile_model(model=model,
                                                                device_name="GPU",))
        for i in range(len(self.compiled_models)):
            self.input_layer_ir.append(self.compiled_models[i].input(0))
            self.output_layer_ir.append(self.compiled_models[i].output(0))
            # print(self.output_layer_ir[i])
        # print('sfjisfshfih')
        # print(self.output_layer_ir[0].get_names)

        # self.input_layer_ir = self.compiled_model.input(0)
        # self.output_layer_ir = self.compiled_model.output("output0")

        # batch size, number of channels, height, width.
        self.batch_size = []
        self.numb_channels = []
        self.height = []
        self.width = []

        self.input_layer_attributes = []


        for i in range(len(self.input_layer_ir)):
            # batch_size,
            # numb_channels,
            # height,
            # width,
            self.input_layer_attributes.append(self.input_layer_ir[i].shape)

        # if isinstance(class_name, list):
        #     self.class_name = tuple(class_name)
        # else:
        #     self.class_name = class_name
        self.class_name = []
        for classes in class_name:
            self.class_name.append(classes)    

    def predict(self, input_image: np.ndarray, model_index: int) -> torch.Tensor:
        """
        Function to use prediction model to return 
        
        :param self: Class for using openvino model
        :param input_image: Image in RGB format
        :type input_image: np.ndarray
        :param model_index: In case of several models used, pick which model to be used
        :type model_index: int
        :return: Returning a list (not python's List) of prediction with format [x1, y1, x2, x2, conf, class]. \n
                 In case of no prediction, returning empty tensoe
        :rtype: Tensor
        """

        
        # print(self.input_layer_attributes[model_index][3], self.input_layer_attributes[model_index][2])
        input_image = preprocess_image_yolov8_format(
            input_image, 
            in_size=(self.input_layer_attributes[model_index][3], self.input_layer_attributes[model_index][2])
        )

        input_image = np.expand_dims(input_image, 0)

        result = self.compiled_models[model_index]([input_image])[self.output_layer_ir[model_index]]

        result = torch.from_numpy(result)

        # result = result.permute(0,2,1) # discovered that yolov8 has different fucking format than previous versions.
        # print(result.shape)
        pred = non_max_suppression(
            result, conf_thres=0.4, iou_thres=0.5, agnostic=False
        )[0]

        # print(pred)
        # clsfy, conf_level, *bbox = reversed(np.array(pred[0]))
        # result = self.compiled_models[model_index]([input_image])[self.output_layer_ir[model_index]]
        # print(result)
        # confidence_level = self.softmax(result)
        # label = np.squeeze(np.argmax(confidence_level, axis=1))

        # return self.class_name[model_index][label]
        # print(clsfy)
        # return self.class_name[model_index][int(clsfy)]
        # print(f'{pred} \n -----------------------------------')
        return pred

    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum()
    

    # # Based on object detection function
    # def predict(self, image, model_index):
    #     input_image = preprocess_image_yolov5_format(
    #         image, in_size=(self.input_layer_attributes[model_index][3],
    #                         self.input_layer_attributes[model_index][2])
    #     )

    #     input_image = np.expand_dims(input_image, 0)

    #     result = self.compiled_models[model_index]([input_image])[self.output_layer_ir]

    #     result = torch.from_numpy(result)

    #     pred = non_max_suppression(
    #         result, conf_thres=0.4, iou_thres=0.5, agnostic=False
    #     )[0]

    #     clsfy, conf_level, *bbox = reversed(np.array(pred[0]))

    #     return self.CLASS[int(clsfy)]