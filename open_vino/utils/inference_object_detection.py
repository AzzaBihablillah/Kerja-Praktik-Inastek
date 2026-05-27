from openvino import Core
from .detector_utils import non_max_suppression, save_output
from .preprocess import preprocess_image_yolov8_format
import torch
import numpy as np


class Detect:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.core = Core()
        self.model = self.core.read_model(model=self.model_path)
        self.compiled_model = self.core.compile_model(
            model=self.model, device_name="CPU"
        )

        self.input_layer_ir = self.compiled_model.input(0)
        self.output_layer_ir = self.compiled_model.output("output0")

        # batch size, number of channels, height, width.
        (
            self.batch_size,
            self.numb_channels,
            self.height,
            self.width,
        ) = self.input_layer_ir.shape

        self.CLASS = ["large", "medium", "small"]

    def predict(self, image):
        input_image = preprocess_image_yolov8_format(
            image, in_size=(self.width, self.height)
        )

        input_image = np.expand_dims(input_image, 0)

        result = self.compiled_model([input_image])[self.output_layer_ir]

        result = torch.from_numpy(result)

        pred = non_max_suppression(
            result, conf_thres=0.4, iou_thres=0.5, agnostic=False
        )[0]

        clsfy, conf_level, *bbox = reversed(np.array(pred[0]))

        # save_output(
        #     pred,
        #     image,
        #     "hasil.jpg",
        #     threshold=0.7,
        #     model_in_HW=(224, 224),
        #     line_thickness=None,
        #     text_bg_alpha=0.0,
        #     CLASS=self.CLASS,
        # )

        return self.CLASS[int(clsfy)]
