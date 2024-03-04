# onnx-quantization
Model quantization and deployment using ONNX runtime

## Requirements

- For the `onnx-quant.ipynb` notebook (obtained through pip): `torch`, `onnx`, `onnxruntime-gpu` (for GPU usage, otherwise `onnxruntime` will do)
- For building the executable for `resnet_inference.cpp` (download pre-built packages for your OS or build from source): `OpenCV`, `onnxruntime`, `cmake`

## Steps

1. The [onnx-quant.ipynb](onnx-quant.ipynb) notebook contains code to train a ResNet18 in PyTorch, quantize it and convert it to ONNX.
2. The [resnet_inference.cpp](resnet_inference.cpp) file contains code to run inference using that ONNX model using the C++ API of ONNX runtime.
3. The [CMakeLists.txt](CMakeLists.txt) file contains the required dependencies to build the executable for `resnet_inference.cpp`.
    - Build steps:   
    ```bash
        (onnx-quantization) $ mkdir build
        (onnx-quantization) $ cd build
        (build) $ cmake ..
        (build) $ make
    ``` 
    - An executable called `classifier` will be created in the `build` directory.
    - Provide the paths to the model file and image file to run inference on as command-line args while running `classifier`. 
    
    
## References

- [ONNX runtime documentation](https://onnxruntime.ai/docs/) 
- [This](https://github.com/leimao/ONNX-Runtime-Inference/) super helpful repository
- [Gemini](https://gemini.google.com/) helped with debugging during the build process xD