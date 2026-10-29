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
./edge-vision.sh -r --model path/to/model.onnx --input path/to/img.jpg  

# build & run, save the image with boxes and the detections
./edge-vision.sh -r --model path/to/model.onnx --input path/to/img.jpg --output out.jpg --detections dets.csv

./edge-vision.sh -t     # build & tests
./edge-vision.sh -d     # build & docs
```

# Model

The model must be a YOLO exported **end-to-end**, with Non-Maximum Suppression (NMS) inside the ONNX graph
(e.g. `models/yolov7-tiny/yolov7-tiny.onnx`). The program does not run NMS itself: it expects the output
to be already filtered, with one row per detection:

```
[batch_id, x0, y0, x1, y1, class_id, score]   # shape [N, 7], coordinates in model input pixels
```

A model exported without NMS (raw `[1, 25200, 85]` output) is rejected with a wrong output shape error.

# Output format

`--detections dets.csv` writes one header line, then one row per detection:

```
class_id,class_name,score,x,y,w,h
2,car,0.7585,441.13,543.64,59.35,47.54
```

| Column | Meaning |
|--------|---------|
| `class_id` | COCO class index (0-79) |
| `class_name` | COCO class name |
| `score` | confidence in [0, 1], 4 decimals |
| `x`, `y` | top-left corner of the box, in original image pixels |
| `w`, `h` | width and height of the box, in original image pixels |

Detections with a score below 0.25 are not written.
