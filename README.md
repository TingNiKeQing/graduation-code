# graduation-code
Graduation_code
Evaluating FRBNet as a Low-Light Enhancement Front-End for DepthAnythingV2
This repository contains the code developed for an MSc dissertation project evaluating whether FRBNet, a frequency-domain low-light feature enhancement network, improves monocular depth estimation with DepthAnythingV2 under low-light and night-time conditions.
The project integrates FRBNet as a plug-in front-end ahead of a frozen DepthAnythingV2 backbone (`low-light RGB → FRBNet → enhanced RGB → DepthAnythingV2 → depth map`), and evaluates it across a baseline-plus-three-stage experimental ladder (no FRBNet → inference-only FRBNet → FRBNet trained on synthetic low light → FRBNet fine-tuned on real night-time driving data), on the NYU Depth V2 and Oxford RobotCar (RobotCarNight) datasets.
File	Role
`darkchange.py`	Physically-motivated Dark-ISP pipeline for synthesising low-light images from well-lit photographs at training time (adapted from MAET, see Acknowledgements)
`train.py`	Stage 2 training: FRBNet initialised from ExDark detection-pretrained weights, trained on NYU Depth V2 with on-the-fly Dark-ISP synthetic low light, supervised by real NYU depth
`fittrain.py`	Stage 3 fine-tuning: loads the Stage 2 checkpoint and continues training on real RobotCarNight image/depth pairs (imports shared model/loss code directly from `train.py`)
`run_depth_only.py`	Baseline-only inference: runs DepthAnythingV2 directly on raw low-light images (no FRBNet), with optional GT scoring (AbsRel, RMSE, δ1, MAE, correlation) written to `summary.csv`
`compare.py`	Produces the quantitative results reported in Chapter 4 of the dissertation. Runs both the baseline and FRBNet-enhanced pipelines on the same images, scores both against ground truth, and sorts each image into `improved/` or `not_improved/` based on the chosen metric. Outputs `summary.csv` (baseline vs. enhanced metrics per frame) plus per-frame `_depth_compare.png` triptychs (baseline / enhanced / GT depth).
`run_pipeline_frbnet_to_depth.py`	Standalone FRBNet → DepthAnythingV2 pipeline runner used to generate illustrative enhanced images and depth maps. Does not compare against a baseline or score against ground truth — see the note below before running it.
