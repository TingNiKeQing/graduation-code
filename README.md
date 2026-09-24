# graduation-code
Graduation_code
Evaluating FRBNet as a Low-Light Enhancement Front-End for DepthAnythingV2
This repository contains the code developed for an MSc dissertation project evaluating whether FRBNet, a frequency-domain low-light feature enhancement network, improves monocular depth estimation with DepthAnythingV2 under low-light and night-time conditions.
The project integrates FRBNet as a plug-in front-end ahead of a frozen DepthAnythingV2 backbone (`low-light RGB → FRBNet → enhanced RGB → DepthAnythingV2 → depth map`), and evaluates it across a baseline-plus-three-stage experimental ladder (no FRBNet → inference-only FRBNet → FRBNet trained on synthetic low light → FRBNet fine-tuned on real night-time driving data), on the NYU Depth V2 and Oxford RobotCar (RobotCarNight) datasets.
## Contents
`darkchange.py`: physically-motivated Dark-ISP pipeline for synthesising low-light images from well-lit photographs
`train.py`: Stage 2 training: FRBNet initialised from ExDark detection-pretrained weights, trained on NYU Depth V2 with on-the-fly Dark-ISP synthetic low light, supervised by real NYU depth
`fittrain.py`:  Stage 3 fine-tuning: loads the Stage 2 checkpoint and continues training on real RobotCarNight image/depth pairs
`run_depth_only.py` : Baseline-only inference: runs DepthAnythingV2 directly on raw low-light images
`run_pipeline_frbnet_to_depth.py`: Standalone FRBNet → DepthAnythingV2 pipeline runner used to generate illustrative enhanced images and depth maps.
## Before You Run `run_pipeline_frbnet_to_depth.py`
Unlike the other scripts, this one does not take model paths as command-line arguments. Several paths are hardcoded near the top of the file for the author's own machine, for example:
```python
FRBNET\_CONFIG = r"D:\\frbnet\\mmdetection\\configs\\yolov3\_frbnet\_exdark.py"
FRBNET\_CHECKPOINT = r"D:\\frbnet\\checkpoint\\frbnet\_stage2\_epoch9.pth"
DEPTH\_CHECKPOINT = r"D:\\frbnet\\Depth-Anything-V2\\checkpoints\\depth\_anything\_v2\_vitb.pth"
```
You must edit these lines directly to point at your own local paths before running this script. Note also that the FRBNet checkpoint hardcoded above (`frbnet\_stage2\_epoch9.pth`) is an intermediate epoch used for illustrative/exploratory output only, and is not the same checkpoint used for the dissertation's quantitative Chapter 4 results
## Setup
```bash
git clone <this-repository-url>
cd <repository-name>
pip install -r requirements.txt
```
This project depends on external code for FRBNet and DepthAnythingV2, which are not vendored into this repository (see Acknowledgements and Licensing below). You will need to clone those repositories separately and point the relevant environment variables / config paths at them:
```bash
git clone https://github.com/Sing-Forevet/FRBNet
git clone https://github.com/DepthAnything/Depth-Anything-V2
export DAV2\_REPO\_PATH=/path/to/Depth-Anything-V2
```
`train.py` and `fittrain.py` additionally require a build of `mmdetection` with FRBNet's detector registered under `mmdet.models.detectors.frbnet\_utils` (i.e. FRBNet's own repository set up as described in its README), since `FIINet` is imported from that module path.
