# graduation-code
Graduation_code
Evaluating FRBNet as a Low-Light Enhancement Front-End for DepthAnythingV2
This repository contains the code developed for an MSc dissertation project evaluating whether FRBNet, a frequency-domain low-light feature enhancement network, improves monocular depth estimation with DepthAnythingV2 under low-light and night-time conditions.
The project integrates FRBNet as a plug-in front-end ahead of a frozen DepthAnythingV2 backbone (`low-light RGB → FRBNet → enhanced RGB → DepthAnythingV2 → depth map`), and evaluates it across a baseline-plus-three-stage experimental ladder (no FRBNet → inference-only FRBNet → FRBNet trained on synthetic low light → FRBNet fine-tuned on real night-time driving data), on the NYU Depth V2 and Oxford RobotCar (RobotCarNight) datasets.
