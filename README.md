# PF-IGEV
[Marrying Polarization to Stereo: Real-Time Stereo Matching via Polarimetric Cues] <br/>
Junzhuo Zhou, Jun Zou, Ye Qiu, Zhihe Liu, Jia Hao, Wenli Li, Yiting Yu  <br/>

## Demo Display
![PF-IGEV Demo](./images/output.gif) <br/>
*Figure 1: Performance demonstration of the PF-IGEV stereo matching algorithm. In the animated image, the first row shows the original left image, the second row displays the polarization fusion result, and the third row presents the stereo matching result.*

## Environment
* NVIDIA RTX 4090
* Python 3.8

### Create a virtual environment and activate it

```shell
conda create -n PFIGEV python=3.8
conda activate PFIGEV
```

### Dependencies

```shell
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

### Dataset Layout

The training and evaluation scripts assume that datasets are stored in a sibling directory named `datasets` relative to the project root. The default paths used in the code are:

* `../datasets/sceneflow/`
* `../datasets/kitti/`
* `../datasets/kitti_pix2pix/` for the polarization-fusion branch

#### SceneFlow

For `train_stereo_rt.py` and `evaluate_stereo_rt.py`, the code expects the SceneFlow directory to follow the standard layout below:

```text
../datasets/sceneflow/
├── frames_finalpass/
│   ├── TRAIN/
│   │   ├── A/
│   │   │   ├── 0000/
│   │   │   │   ├── left/*.png
│   │   │   │   └── right/*.png
│   │   │   └── ...
│   │   ├── B/
│   │   └── C/
│   └── TEST/
│       ├── A/
│       ├── B/
│       └── C/
├── disparity/
│   ├── TRAIN/
│   │   ├── A/
│   │   │   ├── 0000/
│   │   │   │   ├── left/*.pfm
│   │   │   │   └── right/*.pfm
│   │   │   └── ...
│   │   ├── B/
│   │   └── C/
│   └── TEST/
│       ├── A/
│       ├── B/
│       └── C/
```

The loader also reads the Monkaa and Driving subsets from the same root, so keeping the official full SceneFlow directory structure is recommended.

#### KITTI

For the real-time stereo branch, the default KITTI layout is:

```text
../datasets/kitti/
├── 2012/
│   ├── training/
│   │   ├── colored_0/*_10.png
│   │   ├── colored_1/*_10.png
│   │   └── disp_occ/*_10.png
│   └── testing/
│       ├── colored_0/*_10.png
│       └── colored_1/*_10.png
└── 2015/
    ├── training/
    │   ├── image_2/*_10.png
    │   ├── image_3/*_10.png
    │   └── disp_occ_0/*_10.png
    └── testing/
        ├── image_2/*_10.png
        └── image_3/*_10.png
```

For the polarization-fusion branch, the repository expects a second KITTI-style root containing the fused or translated polarization images:

```text
../datasets/kitti_pix2pix/
├── 2012/
│   ├── training/
│   │   ├── colored_0/*_10.png
│   │   └── colored_1/*_10.png
│   └── testing/
│       ├── colored_0/*_10.png
│       └── colored_1/*_10.png
└── 2015/
    ├── training/
    │   ├── image_2/*_10.png
    │   └── image_3/*_10.png
    └── testing/
        ├── image_2/*_10.png
        └── image_3/*_10.png
```

The files in `kitti_pix2pix` must match the filenames and split structure of the original KITTI images so that each fused image can be paired with its corresponding RGB stereo sample.

### `.pth` Checkpoint Files

Pretrained `.pth` files are shared through Quark Cloud:

* Link: https://pan.quark.cn/s/cc4929095388
* Extraction code: `J5sP`

## Free Access to the Paper

The published paper can be accessed through the following personalized Share Link, which provides free access to the final ScienceDirect version before June 11, 2026:

* https://authors.elsevier.com/c/1mzyW3INukZxtq

No registration or payment is required during the active sharing period.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{ZHOU2026133715,
  title    = {Marrying polarization to stereo: Real-time stereo matching via polarimetric cues},
  journal  = {Neurocomputing},
  volume   = {686},
  pages    = {133715},
  year     = {2026},
  issn     = {0925-2312},
  doi      = {https://doi.org/10.1016/j.neucom.2026.133715},
  url      = {https://www.sciencedirect.com/science/article/pii/S0925231226011124},
  author   = {Junzhuo Zhou and Jun Zou and Ye Qiu and Zhihe Liu and Jia Hao and Wenli Li and Yiting Yu},
  keywords = {Stereo matching, Polarization fusion, Cost volume construction, Deep learning, 3D reconstruction}
}
```

## Acknowledgements

This project is based on [IGEV-plusplus](https://github.com/gangweiX/IGEV-plusplus), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [GMStereo](https://github.com/autonomousvision/unimatch), and [CoEx](https://github.com/antabangun/coex). We thank the original authors for their excellent work.


