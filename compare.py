import os
import sys
import shutil
import argparse
import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image
from torchvision import transforms
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
    if valid.sum() < min_valid_points:
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
        "n_valid": int(valid.sum()),
    }
def enhance_with_frbnet(frb_model, pil_img, device):
    x = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced = frb_model.frb_net(x)
    enhanced = enhanced.squeeze(0).cpu()
    flat = enhanced.flatten()
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    enhanced_vis = torch.clamp((enhanced - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    enhanced_pil = transforms.ToPILImage()(enhanced_vis)
    enhanced_pil = enhanced_pil.resize(pil_img.size)
    return enhanced_pil
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
def load_frbnet_checkpoint(frb_model, ckpt_path):
    raw = safe_torch_load(ckpt_path)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw

    has_frb_prefix = any(k.startswith("frb_net.") for k in state_dict.keys())
    if has_frb_prefix:
        missing, unexpected = frb_model.load_state_dict(state_dict, strict=False)
        print(f"checkpoint: {ckpt_path}")
        missing_frb = [k for k in missing if k.startswith("frb_net.")]
        if missing_frb:
            print(f"error1")
    else:
        frb_model.frb_net.load_state_dict(state_dict)
        print(f"fit_checkpoint: {ckpt_path}")
def main():
    parser = argparse.ArgumentParser(description="compare")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--depth-gt-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frbnet-config", required=True)
    parser.add_argument("--depth-ckpt", required=True)
    parser.add_argument("--depth-encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--output-dir", default="./sorted_by_improvement")
    parser.add_argument("--metric", default="abs_rel", choices=["mae", "rmse", "abs_rel", "delta1"])
    parser.add_argument("--thumb-width", type=int, default=280)
    args = parser.parse_args()
    improved_dir = os.path.join(args.output_dir, "improved")
    not_improved_dir = os.path.join(args.output_dir, "not_improved")
    os.makedirs(improved_dir, exist_ok=True)
    os.makedirs(not_improved_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmap = get_colormap("Spectral_r")
    from mmdet.apis import init_detector
    import mmdet.models.detectors.frbnet_utils
    frb_model = init_detector(args.frbnet_config, checkpoint=None, device=device)
    if args.checkpoint is not None:
        load_frbnet_checkpoint(frb_model, args.checkpoint)
        print(f"checkpoint: {args.checkpoint}")
    else:
        print(f"error2")
    frb_model.eval()
    from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    depth_model = DepthAnythingV2(**model_configs[args.depth_encoder])
    depth_model.load_state_dict(torch.load(args.depth_ckpt, map_location="cpu"))
    depth_model = depth_model.to(device).eval()
    image_files = sorted(
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not image_files:
        print(f"error3")
        return
    print(f"all: {len(image_files)} \n")
    HIGHER_IS_BETTER = {"delta1"}
    csv_rows = ["filename,baseline_mae,baseline_rmse,baseline_abs_rel,baseline_delta1,baseline_corr,"
                "enhanced_mae,enhanced_rmse,enhanced_abs_rel,enhanced_delta1,enhanced_corr,"
                "improved,delta_" + args.metric]
    n_improved, n_not_improved, n_skipped = 0, 0, 0
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(args.depth_gt_dir, base_name + ".npy")
        if not os.path.exists(gt_path):
            print(f"error4")
            n_skipped += 1
            continue
        gt_depth = np.squeeze(np.load(gt_path)).astype(np.float32)
        pil_img = Image.open(img_path).convert("RGB")
        orig_rgb = np.array(pil_img)
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        baseline_depth = depth_model.infer_image(orig_bgr, args.depth_input_size)
        score_base = align_and_score(baseline_depth, gt_depth)
        enhanced_pil = enhance_with_frbnet(frb_model, pil_img, device)
        enhanced_rgb = np.array(enhanced_pil)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        enhanced_depth = depth_model.infer_image(enhanced_bgr, args.depth_input_size)
        score_enh = align_and_score(enhanced_depth, gt_depth)
        if score_base is None or score_enh is None:
            print(f"error5")
            n_skipped += 1
            continue
        if args.metric in HIGHER_IS_BETTER:
            improved = score_enh[args.metric] > score_base[args.metric]
        else:
            improved = score_enh[args.metric] < score_base[args.metric]
        delta = score_enh[args.metric] - score_base[args.metric]
        target_dir = improved_dir if improved else not_improved_dir
        status_str = "improve" if improved else "not_improve"
        print(f"[{base_name}] baseline {args.metric}={score_base[args.metric]:.4f}  "
              f"enhanced {args.metric}={score_enh[args.metric]:.4f}  -> {status_str} (Δ={delta:+.4f})")
        if improved:
            n_improved += 1
        else:
            n_not_improved += 1
        shutil.copy(img_path, os.path.join(target_dir, f"{base_name}_original{os.path.splitext(img_path)[1]}"))
        enhanced_pil.save(os.path.join(target_dir, f"{base_name}_enhanced.png"))
        base_vis = make_thumbnail(colorize_depth(baseline_depth, cmap), args.thumb_width)
        enh_vis = make_thumbnail(colorize_depth(enhanced_depth, cmap), args.thumb_width)
        gt_vis = make_thumbnail(colorize_depth(gt_depth, cmap), args.thumb_width)
        base_vis = put_label(base_vis, f"baseline {args.metric}={score_base[args.metric]:.3f}")
        enh_vis = put_label(enh_vis, f"enhanced {args.metric}={score_enh[args.metric]:.3f}")
        gt_vis = put_label(gt_vis, "GT depth")
        min_h = min(base_vis.shape[0], enh_vis.shape[0], gt_vis.shape[0])
        triptych = np.concatenate([base_vis[:min_h], enh_vis[:min_h], gt_vis[:min_h]], axis=1)
        cv2.imwrite(os.path.join(target_dir, f"{base_name}_depth_compare.png"),
                    cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        csv_rows.append(
            f"{base_name},{score_base['mae']:.4f},{score_base['rmse']:.4f},{score_base['abs_rel']:.4f},"
            f"{score_base['delta1']:.4f},{score_base['corr']:.4f},"
            f"{score_enh['mae']:.4f},{score_enh['rmse']:.4f},{score_enh['abs_rel']:.4f},"
            f"{score_enh['delta1']:.4f},{score_enh['corr']:.4f},"
            f"{int(improved)},{delta:.4f}"
        )
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    total_scored = n_improved + n_not_improved
    print("\n===all===")
    print(f"improved: {n_improved} ")
    print(f"not_improved: {n_not_improved} ")
    if total_scored > 0:
        print(f"improve: {n_improved/total_scored*100:.1f}%")
    print(f"\nsave: {args.output_dir}")
    print(f"save: {csv_path}")
if __name__ == "__main__":
    main()