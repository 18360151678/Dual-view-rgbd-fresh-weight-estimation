# YOLO12 segmentation

Download `yolo12_best.pt` and `segmentation_test_set.zip` from the `v2.0`
release. Arrange the files as follows:

```text
segmentation/
  weights/yolo12_best.pt
  test_set/images/*.png
  test_set/labels/*.txt
  evaluate.py
```

Install dependencies and run the test script:

```bash
pip install -r requirements.txt
python evaluate.py
```

The script evaluates all 204 test images and writes official mask mAP,
per-image IoU/Dice, parameter count, FLOPs, latency, and FPS to `output/`.
Paths, device, input size, and batch size can be changed with command-line
arguments; run `python evaluate.py --help` for details.

The released checkpoint is the newer `yolo12.2.0` best checkpoint. Its source
training run used 640-pixel inputs and pretrained initialization.

