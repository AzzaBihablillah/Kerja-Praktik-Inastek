from ultralytics import YOLO

model_path = 'brand_best_ver5.pt'

model = YOLO(model_path)
print(model.export(format='openvino'))