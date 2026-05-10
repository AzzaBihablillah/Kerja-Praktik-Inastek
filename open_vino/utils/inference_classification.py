import numpy as np
from openvino.runtime import Core


class Classify:
    def __init__(self, model_path, class_name):
        self.model_path = model_path
        self.class_name = class_name
        self.core = Core()
        self.model_ir = self.core.read_model(model=self.model_path)
        self.compiled_model_ir = self.core.compile_model(
            model=self.model_ir, device_name="CPU"
        )

        self.output_layer_ir = self.compiled_model_ir.output(0)

    def predict(self, input_image):
        result = self.compiled_model_ir([input_image])[self.output_layer_ir]
        confidence_level = self.softmax(result)
        label = np.squeeze(np.argmax(confidence_level, axis=1))

        return self.class_name[label]

    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum()
