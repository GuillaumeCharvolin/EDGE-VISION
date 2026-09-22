# PROJECT GOAL

Run an object-detection model on each frame on video and measures how long each stage takes.

## Dependency

- OpenCV
- ONNXRuntime
```
ONNXRuntime setup for Ubuntu : 
https://medium.com/@massimilianoriva96/onnxruntime-integration-with-ubuntu-and-cmake-5d7af482136a
```

# Command

```
./edge-vision.sh        # build
  
# build & run
./edge-vision.sh -r --model path/to/model.onnx --image path/to/img.jpg  

./edge-vision.sh -t     # build & tests
./edge-vision.sh -d     # build & docs
```