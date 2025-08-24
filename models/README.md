# models/

Place your ONNX feature-extractor models here. The app will auto-load all `*.onnx` in this folder at startup.

Requirements
- Input: 4D tensor NCHW (batch, 3, H, W). Any H and W are OK; the app auto-resizes.
- Preprocessing: ImageNet mean/std is applied inside the app:
  - mean = [0.485, 0.456, 0.406]
  - std  = [0.229, 0.224, 0.225]
- Output: Any shape. The app flattens and L2-normalizes the first output tensor as an embedding.

Recommended models
- ResNet50 (resnet50-v2-7.onnx)
- MobileNetV2 (mobilenetv2-7.onnx)

Quick add (macOS/zsh)
1) From the project root, run:

   curl -L -o models/resnet50-v2-7.onnx \
     https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

   curl -L -o models/mobilenetv2-7.onnx \
     https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

2) Restart the app. You should see a status like: `Deep: 2 โมเดล (tag XXXXX)`

Custom export (PyTorch)
- Ensure your model expects 3xHxW input, then:

  torch.onnx.export(
    model.eval(),
    torch.randn(1,3,224,224),
    "models/my_model.onnx",
    input_names=["input"],
    output_names=["feat"],
    opset_version=12,
    dynamic_axes={"input": {0: "batch"}, "feat": {0: "batch"}}
  )

Notes
- Place multiple ONNX files to enable an ensemble; embeddings are concatenated and L2-normalized.
- The model tag is derived from filenames; changing files will change the tag (and classifier file name).
