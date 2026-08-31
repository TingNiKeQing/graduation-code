import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image
from torchvision import transforms
import sys
import os as _os
_dav2_path = _os.environ.get("DAV2_REPO_PATH", r"D:\frbnet\Depth-Anything-V2")
sys.path.append(_dav2_path)
from mmdet.apis import init_detector
from depth_anything_v2.dpt import DepthAnythingV2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRBNET_CONFIG = r"D:\frbnet\mmdetection\configs\yolov3_frbnet_exdark.py"
FRBNET_CHECKPOINT = r"D:\frbnet\checkpoint\frbnet_stage2_epoch9.pth"
DEPTH_ENCODER = "vitb"
DEPTH_CHECKPOINT = r"D:\frbnet\Depth-Anything-V2\checkpoints\depth_anything_v2_vitb.pth"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
def enhance_one_image(frb_model, input_path, output_path, device):
    img = Image.open(input_path).convert("RGB")
    orig_size = img.size
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced = frb_model.frb_net(x)
    enhanced = enhanced.squeeze(0).cpu()
    enh_min, enh_max = enhanced.min(), enhanced.max()
    enhanced_vis = (enhanced - enh_min) / (enh_max - enh_min + 1e-8)
    enhanced_pil = transforms.ToPILImage()(enhanced_vis)
    enhanced_pil = enhanced_pil.resize(orig_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enhanced_pil.save(output_path)
def run_enhance_stage(frb_model, input_dir, output_dir, device):
    print(f"FRBNet enhance:{input_dir} -> {output_dir}")
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                image_files.append(os.path.join(root, f))
    if not image_files:
        print("error1")
        return []
    output_paths = []
    for i, input_path in enumerate(sorted(image_files), 1):
        rel_path = os.path.relpath(input_path, input_dir)
        name_without_ext = os.path.splitext(rel_path)[0]
        output_path = os.path.join(output_dir, name_without_ext + "_enhanced.png")
        try:
            enhance_one_image(frb_model, input_path, output_path, device)
            output_paths.append(output_path)
        except Exception as e:
            print(f"error2")
        if i % 20 == 0 or i == len(image_files):
            print(f"FRBnet:{i}/{len(image_files)}")
    print(f"FRBNet success:{len(output_paths)} \n")
    return output_paths
def run_depth_stage(depth_model, input_dir, output_dir, input_size=518,
                     npy_suffix="_enhanced_raw_depth.npy", grayscale=False):
    print(f"DepthAnythingV2：{input_dir} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                image_files.append(os.path.join(root, f))
    if not image_files:
        print(f"error3")
        return
    for i, filename in enumerate(sorted(image_files), 1):
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"error4")
            continue
        raw_depth = depth_model.infer_image(raw_image, input_size)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        npy_path = os.path.join(output_dir, base_name + npy_suffix)
        np.save(npy_path, raw_depth)
        summary_path = os.path.join(output_dir, base_name + "_depth_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"shape: {raw_depth.shape}\n")
            f.write(f"min depth: {float(raw_depth.min())}\n")
            f.write(f"max depth: {float(raw_depth.max())}\n")
            f.write(f"mean depth: {float(raw_depth.mean())}\n")
            f.write(f"std depth: {float(raw_depth.std())}\n")
        vis_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-8) * 255.0
        vis_depth = vis_depth.astype(np.uint8)
        if grayscale:
            vis_depth = np.repeat(vis_depth[..., np.newaxis], 3, axis=-1)
        else:
            vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        vis_path = os.path.join(output_dir, base_name + "_depth_vis.png")
        cv2.imwrite(vis_path, vis_depth)
        if i % 20 == 0 or i == len(image_files):
            print(f"DepthAnythingV2:{i}/{len(image_files)}")

    print(f" success:{output_dir}\n")
def main():
    parser = argparse.ArgumentParser(description="FRBNet -> DepthAnythingV2 ")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--enhanced-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()
    frb_model = init_detector(FRBNET_CONFIG, checkpoint=None, device=DEVICE)
    frb_model.frb_net.load_state_dict(torch.load(FRBNET_CHECKPOINT, map_location="cpu"))
    frb_model.eval()
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    depth_model = DepthAnythingV2(**model_configs[DEPTH_ENCODER])
    depth_model.load_state_dict(torch.load(DEPTH_CHECKPOINT, map_location="cpu"))
    depth_model = depth_model.to(DEVICE).eval()
    run_enhance_stage(frb_model, args.input_dir, args.enhanced_dir, DEVICE)
    run_depth_stage(depth_model, args.enhanced_dir, args.depth_dir,
                     input_size=args.input_size,
                     npy_suffix="_enhanced_raw_depth.npy",
                     grayscale=args.grayscale)
if __name__ == "__main__":
    main()