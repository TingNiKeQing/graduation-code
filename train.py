import os
import sys
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
_dav2_path = os.environ.get("DAV2_REPO_PATH", r"D:\frbnet\Depth-Anything-V2")
sys.path.append(_dav2_path)
from mmdet.models.detectors.frbnet_utils import FIINet
from depth_anything_v2.dpt import DepthAnythingV2
from darkchange import DarkISP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
def letterbox_resize(img, target_size, interpolation, pad_value=0):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    if img.ndim == 3:
        padded = np.full((target_size, target_size, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        padded = np.full((target_size, target_size), pad_value, dtype=img.dtype)
    padded[top:top + new_h, left:left + new_w] = resized

    pad_info = {"top": top, "left": left, "new_h": new_h, "new_w": new_w,
                "orig_h": h, "orig_w": w, "scale": scale}
    return padded, pad_info
class NYUDayGTDataset(Dataset):
    EIGEN_CROP = (45, 471, 41, 601)
    def __init__(self, data_root, split_file=None, img_size=518,
                 min_gt=1e-3, max_gt=10.0, depth_scale=1000.0, use_eigen_crop=True,
                 use_dark_isp=True, dark_isp_kwargs=None):
        self.data_root = data_root
        self.img_size = img_size
        self.min_gt = min_gt
        self.max_gt = max_gt
        self.depth_scale = depth_scale
        self.use_eigen_crop = use_eigen_crop
        self.use_dark_isp = use_dark_isp
        self.dark_isp = DarkISP(**(dark_isp_kwargs or {})) if use_dark_isp else None
        self.pairs = []
        if split_file is not None:
            missing = 0
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    rgb_rel, depth_rel = parts[0], parts[1]
                    rgb_path = os.path.join(data_root, rgb_rel)
                    depth_path = os.path.join(data_root, depth_rel)
                    if os.path.exists(rgb_path) and os.path.exists(depth_path):
                        self.pairs.append((rgb_path, depth_path))
                    else:
                        missing += 1
            if missing:
                print(f"error")
        else:
            rgb_pattern = os.path.join(data_root, "*", "rgb_*.jpg")
            for rgb_path in sorted(glob.glob(rgb_pattern)):
                fname = os.path.basename(rgb_path)
                depth_fname = fname.replace("rgb_", "sync_depth_").replace(".jpg", ".png")
                depth_path = os.path.join(os.path.dirname(rgb_path), depth_fname)
                if os.path.exists(depth_path):
                    self.pairs.append((rgb_path, depth_path))
        assert len(self.pairs) > 0, (
            f"error2"
        )
        print(f"[NYUDayGTDataset] success:{len(self.pairs)} ")
    def __len__(self):
        return len(self.pairs)
    def _maybe_eigen_crop(self, arr):
        if not self.use_eigen_crop:
            return arr
        h, w = arr.shape[:2]
        if (h, w) != (480, 640):
            if not getattr(self, "_warned_crop_skip", False):
                print(f"error3")
                self._warned_crop_skip = True
            return arr
        top, bottom, left, right = self.EIGEN_CROP
        return arr[top:bottom, left:right]
    def _load_gt_depth(self, depth_path):
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"error4")
        depth_m = depth_raw.astype(np.float32) / self.depth_scale
        depth_m = self._maybe_eigen_crop(depth_m)
        depth_m, _pad_info = letterbox_resize(
            depth_m, self.img_size, interpolation=cv2.INTER_NEAREST, pad_value=0.0
        )
        valid_mask = ((depth_m > self.min_gt) & (depth_m <= self.max_gt)).astype(np.float32)
        return depth_m, valid_mask
    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]
        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            raise FileNotFoundError(f"error5")
        img_bgr = self._maybe_eigen_crop(img_bgr)
        img_bgr, _pad_info = letterbox_resize(
            img_bgr, self.img_size, interpolation=cv2.INTER_LINEAR, pad_value=0
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.dark_isp is not None:
            net_input_rgb, _dark_info = self.dark_isp(img_rgb)
        else:
            net_input_rgb = img_rgb
        gt_depth, valid_mask = self._load_gt_depth(depth_path)
        net_input_t = torch.from_numpy(net_input_rgb.transpose(2, 0, 1)).float()
        clean_ref_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        gt_depth_t = torch.from_numpy(gt_depth).float()
        valid_mask_t = torch.from_numpy(valid_mask).float()
        return net_input_t, clean_ref_t, gt_depth_t, valid_mask_t
def to_valid_image(x, eps=1e-6, lo_q=0.01, hi_q=0.99):
    B = x.shape[0]
    x_flat = x.reshape(B, -1)
    x_flat32 = x_flat.float()
    x_lo = torch.quantile(x_flat32, lo_q, dim=1, keepdim=True).view(B, 1, 1, 1).to(x.dtype)
    x_hi = torch.quantile(x_flat32, hi_q, dim=1, keepdim=True).view(B, 1, 1, 1).to(x.dtype)
    return torch.clamp((x - x_lo) / (x_hi - x_lo + eps), 0.0, 1.0)
def normalize_for_depth_model(img01, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (img01 - mean) / std
def median_mad_align(pred, gt, mask, min_valid_points=20, eps=1e-6):
    B, H, W = pred.shape
    aligned = torch.zeros_like(pred)
    valid_sample = torch.zeros(B, dtype=torch.bool, device=pred.device)
    for b in range(B):
        m = mask[b] > 0.5
        if m.sum() < min_valid_points:
            aligned[b] = pred[b]
            continue
        p = pred[b][m]
        g = gt[b][m]
        t_p = p.median()
        s_p = (p - t_p).abs().mean().clamp(min=eps)
        t_g = g.median()
        s_g = (g - t_g).abs().mean().clamp(min=eps)
        aligned[b] = (pred[b] - t_p) / s_p * s_g + t_g
        valid_sample[b] = True
    return aligned, valid_sample
def masked_depth_consistency_loss(pred, gt, mask, min_valid_points=20, eps=1e-6):
    aligned_pred, valid_sample = median_mad_align(pred, gt, mask, min_valid_points, eps)
    diff = torch.abs(aligned_pred - gt) * mask
    n = mask.sum(dim=(1, 2)).clamp(min=1.0)
    per_sample_loss = diff.sum(dim=(1, 2)) / n
    if valid_sample.sum() == 0:
        loss = (per_sample_loss * 0).mean()
    else:
        loss = per_sample_loss[valid_sample].mean()
    return loss, aligned_pred
def edge_aware_smoothness_loss(depth, image):
    depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    img_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)
    return (depth_dx * weight_x).mean() + (depth_dy * weight_y).mean()
def ssim_loss(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    pad = window_size // 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()
def fidelity_loss(enhanced01, clean, weight=0.1, ssim_weight=1.0):
    l1 = F.l1_loss(enhanced01, clean)
    ssim = ssim_loss(enhanced01, clean)
    return weight * (l1 + ssim_weight * ssim)
def build_frbnet(number_K=10, lamda=0.1, pretrained_det_ckpt=None):
    frbnet = FIINet(number_K=number_K, lamda=lamda)
    if pretrained_det_ckpt is not None and os.path.exists(pretrained_det_ckpt):
        ckpt = torch.load(pretrained_det_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        frb_state = {
            k[len("frb_net."):]: v
            for k, v in state_dict.items() if k.startswith("frb_net.")
        }
        missing, unexpected = frbnet.load_state_dict(frb_state, strict=False)
        print(f"[FIINet warm start]loaded:{len(frb_state)} ，"
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    return frbnet
def train(
    data_root,
    split_file=None,
    save_dir="./checkpoints",
    epochs=20,
    batch_size=4,
    lr=1e-4,
    lambda_smooth=0.05,
    lambda_fidelity=1.0,
    min_gt=1e-3,
    max_gt=10.0,
    depth_scale=1000.0,
    use_eigen_crop=True,
    use_dark_isp=True,
    dark_isp_kwargs=None,
    frbnet_pretrained_det_ckpt=None,
    depth_encoder="vitb",
    depth_ckpt_path=r"D:\frbnet\Depth-Anything-V2\checkpoints\depth_anything_v2_vitb.pth",
    max_steps=None,
):
    os.makedirs(save_dir, exist_ok=True)
    frbnet = build_frbnet(number_K=10, lamda=0.1, pretrained_det_ckpt=frbnet_pretrained_det_ckpt)
    frbnet.to(DEVICE).train()
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    depth_model = DepthAnythingV2(**model_configs[depth_encoder])
    depth_model.load_state_dict(torch.load(depth_ckpt_path, map_location="cpu"))
    depth_model.to(DEVICE).eval()
    for p in depth_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(frbnet.parameters(), lr=lr)
    dataset = NYUDayGTDataset(
        data_root, split_file=split_file, min_gt=min_gt, max_gt=max_gt,
        depth_scale=depth_scale, use_eigen_crop=use_eigen_crop,
        use_dark_isp=use_dark_isp, dark_isp_kwargs=dark_isp_kwargs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                         drop_last=True, pin_memory=True, persistent_workers=True,
                         worker_init_fn=worker_init_fn)
    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("start(AMP)")
    for epoch in range(epochs):
        epoch_loss = 0.0
        step = -1
        for step, (net_input, clean_ref, gt_depth, valid_mask) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                print(f"success max_steps={max_steps}")
                break
            net_input = net_input.to(DEVICE)
            clean_ref = clean_ref.to(DEVICE)
            gt_depth = gt_depth.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                raw_enhanced = frbnet(net_input)
                enhanced01 = to_valid_image(raw_enhanced)
                enhanced_for_depth = normalize_for_depth_model(enhanced01, DEVICE)
                depth_pred = depth_model(enhanced_for_depth)
                loss_consist, aligned_pred = masked_depth_consistency_loss(depth_pred, gt_depth, valid_mask)
                loss_smooth = edge_aware_smoothness_loss(aligned_pred.unsqueeze(1), enhanced01)
                loss_fid = fidelity_loss(enhanced01, clean_ref, weight=lambda_fidelity)
                loss = loss_consist + lambda_smooth * loss_smooth + loss_fid
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if step % 20 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Step {step}/{len(loader)} "
                      f"loss={loss.item():.4f} (consist={loss_consist.item():.4f}, "
                      f"smooth={loss_smooth.item():.4f}, fid={loss_fid.item():.4f})")
        actual_steps = min(step + 1, max_steps) if max_steps is not None else len(loader)
        print(f"==> Epoch {epoch+1} ave_loss: {epoch_loss/actual_steps:.4f}")
        ckpt_path = os.path.join(save_dir, f"frbnet_epoch{epoch+1}.pth")
        torch.save(frbnet.state_dict(), ckpt_path)
        print(f"save: {ckpt_path}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FRBNet")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--save-dir", default="./checkpoints_frbnet_trained_bright")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-fidelity", type=float, default=1.0)
    parser.add_argument("--min-gt", type=float, default=1e-3)
    parser.add_argument("--max-gt", type=float, default=10.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--no-eigen-crop", action="store_true")
    parser.add_argument("--no-dark-isp", action="store_true")
    parser.add_argument("--darkness", nargs=2, type=float, default=(0.05, 0.15))
    parser.add_argument("--dark-gamma", nargs=2, type=float, default=(2.0, 3.5))
    parser.add_argument("--dark-noise-scale", type=float, default=1.0)
    parser.add_argument("--dark-fix-cast", type=float, default=0.6)
    parser.add_argument("--frbnet-ckpt", default=None)
    parser.add_argument("--depth-encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-ckpt", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    dark_isp_kwargs = {
        "darkness_range": tuple(args.darkness),
        "gamma_range": tuple(args.dark_gamma),
        "noise_scale": args.dark_noise_scale,
        "color_cast_guard": args.dark_fix_cast,
    }
    train(
        data_root=args.data_root,
        split_file=args.split_file,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_smooth=args.lambda_smooth,
        lambda_fidelity=args.lambda_fidelity,
        min_gt=args.min_gt,
        max_gt=args.max_gt,
        depth_scale=args.depth_scale,
        use_dark_isp=not args.no_dark_isp,
        dark_isp_kwargs=dark_isp_kwargs,
        use_eigen_crop=not args.no_eigen_crop,
        frbnet_pretrained_det_ckpt=args.frbnet_ckpt,
        depth_encoder=args.depth_encoder,
        depth_ckpt_path=args.depth_ckpt,
        max_steps=args.max_steps,
    )