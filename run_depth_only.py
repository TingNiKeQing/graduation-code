import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image
_dav2_path = os.environ.get("DAV2_REPO_PATH", r"D:\frbnet\Depth-Anything-V2")
sys.path.append(_dav2_path)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
def get_colormap(name):
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, TypeError, KeyError):
        return matplotlib.cm.get_cmap(name)
def colorize_depth(raw_depth, cmap):
    vis = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-8) * 255.0
    vis = vis.astype(np.uint8)
    vis = (cmap(vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
def make_thumbnail(img_rgb_uint8, thumb_w=320):
    h, w = img_rgb_uint8.shape[:2]
    thumb_h = int(h * thumb_w / w)
    return cv2.resize(img_rgb_uint8, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
def put_label(img, text):
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(labeled, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled
def align_and_score(raw_depth, gt_depth, min_valid_points=20, min_gt=1e-6, delta_threshold=1.25):
    valid = gt_depth > min_gt
    n_valid = int(valid.sum())
    if n_valid < min_valid_points:
        return None
    p = raw_depth[valid]
    g = gt_depth[valid]
    t_p, t_g = np.median(p), np.median(g)
    s_p = np.mean(np.abs(p - t_p)) + 1e-8
    s_g = np.mean(np.abs(g - t_g)) + 1e-8
    aligned = (raw_depth - t_p) / s_p * s_g + t_g
    ap = aligned[valid]
    diff = ap - g
    ap_safe = np.clip(ap, 1e-3, None)
    g_safe = np.clip(g, 1e-3, None)
    ratio = np.maximum(ap_safe / g_safe, g_safe / ap_safe)
    delta1 = float(np.mean(ratio < delta_threshold))
    corr = float(np.corrcoef(p, g)[0, 1]) if len(p) > 1 else 0.0
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "abs_rel": float(np.mean(np.abs(diff) / (g + 1e-8))),
        "delta1": delta1,
        "corr": corr,
        "n_valid": n_valid,
    }
def main():
    parser = argparse.ArgumentParser(description="DepthAnythingV2")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--depth-gt-dir", default=None)
    parser.add_argument("--depth-ckpt", required=True)
    parser.add_argument("--depth-encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--output-dir", default="./depth_only_out")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmap = get_colormap("Spectral_r")
    from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    depth_model = DepthAnythingV2(**model_configs[args.depth_encoder])
    depth_model.load_state_dict(torch.load(args.depth_ckpt, map_location="cpu"))
    depth_model = depth_model.to(device).eval()
    only_ids = None
    if args.only_ids:
        only_ids = set(s.strip() for s in args.only_ids.split(","))
    image_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if only_ids is not None:
        image_files = [f for f in image_files if os.path.splitext(f)[0] in only_ids]
        found_ids = {os.path.splitext(f)[0] for f in image_files}
        missing = only_ids - found_ids
        if missing:
            print(f"error1")
    if not image_files:
        print(f"error2")
        return
    print(f"all:{len(image_files)}\n")
    csv_rows = ["filename,n_valid_gt,mae,rmse,abs_rel,delta1,corr"]
    for fname in image_files:
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(args.input_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")
        orig_rgb = np.array(pil_img)
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        raw_depth = depth_model.infer_image(orig_bgr, args.depth_input_size)
        np.save(os.path.join(args.output_dir, f"{base_name}_raw_depth.npy"), raw_depth)
        depth_vis = colorize_depth(raw_depth, cmap)
        score = None
        gt_depth = None
        gt_path = None
        if args.depth_gt_dir:
            gt_path = os.path.join(args.depth_gt_dir, base_name + ".npy")
            if os.path.exists(gt_path):
                gt_depth = np.squeeze(np.load(gt_path)).astype(np.float32)
                score = align_and_score(raw_depth, gt_depth)
        if score is not None:
            print(f"[{base_name}] GT={score['n_valid']}  "
                  f"AbsRel={score['abs_rel']:.4f}  delta1={score['delta1']:.4f}  corr={score['corr']:+.4f}")
            csv_rows.append(f"{base_name},{score['n_valid']},{score['mae']:.4f},{score['rmse']:.4f},"
                             f"{score['abs_rel']:.4f},{score['delta1']:.4f},{score['corr']:.4f}")
        elif gt_path is not None:
            if gt_depth is None:
                print(f"error3")
            else:
                n_valid = int((gt_depth > 1e-6).sum())
                print(f"error4")
                csv_rows.append(f"{base_name},{n_valid},,,,,")
        else:
            print(f"error5")
        orig_thumb = make_thumbnail(orig_rgb, args.thumb_width)
        depth_thumb = make_thumbnail(depth_vis, args.thumb_width)
        orig_thumb = put_label(orig_thumb, "original")
        label_text = "depth (raw, no enhancement)"
        if score is not None:
            label_text += f"  AbsRel={score['abs_rel']:.3f} corr={score['corr']:+.3f}"
        depth_thumb = put_label(depth_thumb, label_text)
        panels = [orig_thumb, depth_thumb]
        if gt_depth is not None:
            gt_vis = colorize_depth(gt_depth, cmap)
            gt_thumb = make_thumbnail(gt_vis, args.thumb_width)
            gt_thumb = put_label(gt_thumb, "GT depth")
            panels.append(gt_thumb)
        min_h = min(p.shape[0] for p in panels)
        combo = np.concatenate([p[:min_h] for p in panels], axis=1)
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_compare.png"),
                    cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    if args.depth_gt_dir:
        csv_path = os.path.join(args.output_dir, "summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_rows) + "\n")
    print(f"save: {args.output_dir}")
if __name__ == "__main__":
    main()
