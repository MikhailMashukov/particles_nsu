# particles_nsu
Public repository with core code for counting particles by means of neural networks.

### Requirements
(determined by used mmdetection)

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdetection](https://raw.githubusercontent.com/open-mmlab/mmdetection/master/docs/install.md).

You also need to download epoch_500_3x.pth from http://particlesnn.nsu.ru/data/static/weights/epoch_500_3x.pth before start. And to put it into a "weights" subfolder.

The code that works at http://particlesnn.nsu.ru/, MMDetection NanoPart v. 3.0 model is at the nano_predict.py, processFileForSite3_0 procedure. It runs pre-trained neural model, calculates and saves particles statistics. The model initialization code itself is at the getModel3_0 function.
