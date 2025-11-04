# PF-IGEV
[Marrying Polarization to Stereo: Real-Time Stereo Matching via Polarimetric Cues] <br/>
Junzhuo Zhou, Jun Zou, Ye Qiu, Zhihe Liu, Jia Hao, Wenli Li, Yiting Yu  <br/>

## Demo Display
![PF-IGEV算法效果演示](./images/output.gif) <br/>
*Figure 1: Performance demonstration of the PF-IGEV stereo matching algorithm. In the animated image, the first row shows the original left image, the second row displays the polarization fusion result, and the third row presents the stereo matching result.*

## Environment
* NVIDIA RTX 4090
* python 3.8

### Create a virtual environment and activate it.

```Shell
conda create -n PFIGEV python=3.8
conda activate PFIGEV
```
### Dependencies

```Shell
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.5.4

```

## Required Data

* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

## Citation

If you find our works useful in your research, please consider citing our papers:

```bibtex

```

# Acknowledgements

This project is based on [IGEV-plusplus](https://github.com/gangweiX/IGEV-plusplus), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [GMStereo](https://github.com/autonomousvision/unimatch), and [CoEx](https://github.com/antabangun/coex). We thank the original authors for their excellent works.
