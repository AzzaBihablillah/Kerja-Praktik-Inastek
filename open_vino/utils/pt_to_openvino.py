import torch
import os
import glob

ROOT = os.getcwd()

files = glob.glob(os.path.join(ROOT, "*b1b2.pth"))

for model_path in files:
    model_name = model_path.split("/")[-1].split(".")[0]

    dummy_input = torch.randn(1, 3, 224, 224)

    model = torch.load(model_path, map_location=torch.device("cpu"))

    model.eval()

    output_dir = f"models/{model_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(output_dir, f"{model_name}.onnx"),
        opset_version=11,
        do_constant_folding=False,
    )

    input_model = os.path.join(output_dir, f"{model_name}.onnx")
    os.system(
        f'mo --input_model {input_model} --output_dir {output_dir} --input_shape "[1,3,224,224]" --compress_to_fp16=False'
    )
