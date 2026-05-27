from typing import List
from openvino import Core
from .detector_utils import non_max_suppression, save_output
from .preprocess import preprocess_image_yolov8_format, letterbox, pad_to_square, preserve_aspect_ratio_resize
from .postprocess import inverse_letterbox
import torch
import numpy as np

from typing import List, Union
from openvino import Core
import torch
import numpy as np

from .detector_utils import non_max_suppression
from .preprocess import preprocess_image_yolov8_format

MODEL_DETECTION_TYPE = 0
MODEL_CLASSIFICATION_TYPE = 1
MODEL_SEGMENTATION_TYPE = 2


class Models:
    """
    Manages and runs multiple OpenVINO models (detection, classification, segmentation).
    All models are expected to follow the YOLOv8 output format.
    """

    def __init__(
        self,
        model_paths: Union[str, List[str]],
        class_names: List[List[str]],
        types: Union[List[int], int] = MODEL_DETECTION_TYPE,
        device: str = "CPU",
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
    ) -> None:
        """
        :param model_paths: Path or list of paths to OpenVINO .xml model files.
        :param class_names: List of class name lists, one per model.
        :param types: int or List of int specifying the model. Default: detection. Note: only detection is currently available.
        :param device: OpenVINO device string, e.g. "CPU", "GPU", "AUTO".
        :param conf_thres: Confidence threshold for NMS.
        :param iou_thres: IoU threshold for NMS.
        """
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        if isinstance(types, int):
            types = [types]
            for i in range(len(model_paths)-1):
                types.append(types[0])

        if len(model_paths) != len(class_names) and len(model_paths) != len(types):
            raise ValueError(
                f"{len(model_paths)} model path(s) given but class names for "
                f"{len(class_names)} model(s) provided. These must match."
            )

        self.model_paths = model_paths
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        core = Core()

        try:
            self.compiled_models = [
                core.compile_model(model=core.read_model(path), device_name=device)
                for path in model_paths
            ]
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile model on device '{device}'. "
                f"Available devices: {core.available_devices}. Original error: {e}"
            )

        self.input_layers = [m.input(0) for m in self.compiled_models]
        self.output_layers = [m.output(0) for m in self.compiled_models]

        # Validate and cache input shapes — expected: [batch, channels, H, W]
        self.input_shapes = []
        for i, layer in enumerate(self.input_layers):
            shape = layer.shape
            if len(shape) != 4:
                raise ValueError(
                    f"Model {i}: expected 4D input [batch, channels, H, W], "
                    f"got shape {list(shape)}"
                )
            self.input_shapes.append(shape)


    def predict(self, image: np.ndarray, model_index: int = 0) -> Union[np.ndarray, None]:
        """
        Run inference on a single BGR image using the specified model.

        :param image: Input image in BGR format (as returned by cv2.imread).
        :param model_index: Index of the model to use. Default = 0.
        :return: Tensor of shape (n, 6) with columns [x1, y1, x2, y2, conf, class_id],
                 or None if no detections.
        """
        if not 0 <= model_index < len(self.compiled_models):
            raise IndexError(
                f"model_index {model_index} is out of range. "
                f"{len(self.compiled_models)} model(s) loaded."
            )

        img_h, img_w = image.shape[:2]
        _, _, h, w = self.input_shapes[model_index]
        preprocessed = letterbox(image, target_size=w, pad_value=114)
        preprocessed = preprocess_image_yolov8_format(preprocessed, in_size=(w, h))
        preprocessed = np.expand_dims(preprocessed, 0)

        raw = self.compiled_models[model_index]([preprocessed])[self.output_layers[model_index]]
        result = torch.from_numpy(raw)

        detections = non_max_suppression(
            result,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=False,
        )[0].numpy()
        # print(detections)

        if detections.shape[0] != 0:
            # for i in range(len(detections)):
                detections[:4] = inverse_letterbox(detections[:4], (img_h,img_w), w)
        else:
            return None

        return detections

    def get_class_name(self, model_index: int, class_id: int) -> str:
        """Resolve a class ID to its label string for a given model."""
        return self.class_names[model_index][class_id]


# class Models:
#     def __init__(self, model_path, class_name) -> None:
#         if isinstance(model_path, str):
#             self.model_path = (model_path)
#         elif isinstance(model_path, list):
#             self.model_path = tuple(model_path)
#         else:
#             self.model_path = model_path

#         if len(self.model_path) != len(class_name):
#             raise Exception(f'{len(self.model_path)} model(s) was given but class name for {len(class_name)} model(s) was given')
        
#         self.core = Core()
#         self.models = []
#         self.compiled_models = []
#         self.input_layer_ir = []
#         self.output_layer_ir = []
#         # print(self.model_path)
#         for model in self.model_path:
            
#             self.models.append(self.core.read_model(model=model))
#         for model in self.models:
#             self.compiled_models.append(self.core.compile_model(model=model,
#                                                                 device_name="GPU",))
#         for i in range(len(self.compiled_models)):
#             self.input_layer_ir.append(self.compiled_models[i].input(0))
#             self.output_layer_ir.append(self.compiled_models[i].output(0))
#             # print(self.output_layer_ir[i])
#         # print('sfjisfshfih')
#         # print(self.output_layer_ir[0].get_names)

#         # self.input_layer_ir = self.compiled_model.input(0)
#         # self.output_layer_ir = self.compiled_model.output("output0")

#         # batch size, number of channels, height, width.
#         self.batch_size = []
#         self.numb_channels = []
#         self.height = []
#         self.width = []

#         self.input_layer_attributes = []


#         for i in range(len(self.input_layer_ir)):
#             # batch_size,
#             # numb_channels,
#             # height,
#             # width,
#             self.input_layer_attributes.append(self.input_layer_ir[i].shape)

#         # if isinstance(class_name, list):
#         #     self.class_name = tuple(class_name)
#         # else:
#         #     self.class_name = class_name
#         self.class_name = []
#         for classes in class_name:
#             self.class_name.append(classes)    

#     def predict(self, input_image: np.ndarray, model_index: int) -> torch.Tensor:
#         """
#         Function to use prediction model to return 
        
#         :param self: Class for using openvino model
#         :param input_image: Image in RGB format
#         :type input_image: np.ndarray
#         :param model_index: In case of several models used, pick which model to be used
#         :type model_index: int
#         :return: Returning a list (not python's List) of prediction with format [x1, y1, x2, x2, conf, class]. \n
#                  In case of no prediction, returning empty tensoe
#         :rtype: Tensor
#         """

        
#         # print(self.input_layer_attributes[model_index][3], self.input_layer_attributes[model_index][2])
#         input_image = preprocess_image_yolov8_format(
#             input_image, 
#             in_size=(self.input_layer_attributes[model_index][3], self.input_layer_attributes[model_index][2])
#         )

#         input_image = np.expand_dims(input_image, 0)

#         result = self.compiled_models[model_index]([input_image])[self.output_layer_ir[model_index]]

#         result = torch.from_numpy(result)

#         # result = result.permute(0,2,1) # discovered that yolov8 has different fucking format than previous versions.
#         # print(result.shape)
#         pred = non_max_suppression(
#             result, conf_thres=0.4, iou_thres=0.5, agnostic=False
#         )[0]

#         # print(pred)
#         # clsfy, conf_level, *bbox = reversed(np.array(pred[0]))
#         # result = self.compiled_models[model_index]([input_image])[self.output_layer_ir[model_index]]
#         # print(result)
#         # confidence_level = self.softmax(result)
#         # label = np.squeeze(np.argmax(confidence_level, axis=1))

#         # return self.class_name[model_index][label]
#         # print(clsfy)
#         # return self.class_name[model_index][int(clsfy)]
#         # print(f'{pred} \n -----------------------------------')
#         return pred

#     def softmax(self, x):
#         return np.exp(x) / np.exp(x).sum()
    

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