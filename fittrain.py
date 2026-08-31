import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from train import (
        FIINet,
        DepthAnythingV2,
        DEVICE,
        to_valid_image,
        normalize_for_depth_model,
        masked_depth_consistency_loss,
        edge_aware_smoothness_loss,
        fidelity_loss,
        letterbox_resize,
    )
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
def load_frbnet_flexible(frbnet, ckpt_path):
    raw = safe_torch_load(ckpt_path)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw
    has_frb_prefix = any(k.startswith("frb_net.") for k in state_dict.keys())
    if has_frb_prefix:
        frb_state = {
            k[len("frb_net."):]: v
            for k, v in state_dict.items() if k.startswith("frb_net.")
        }
        missing, unexpected = frbnet.load_state_dict(frb_state, strict=False)
        if len(frb_state) == 0:
            print(f"error1")
    else:
        frbnet.load_state_dict(state_dict)
class RealNightGTDataset(Dataset):
    def __init__(self, image_root, depth_root, img_size=518, min_gt=1e-6, max_gt=80.0, exts=(".jpg", ".png", ".jpeg")):
        self.depth_root = depth_root
        self.img_size = img_size
        self.min_gt = min_gt
        self.max_gt = max_gt
        self.image_paths = []
        for ext in exts:
            self.image_paths += glob.glob(os.path.join(image_root, f"**/*{ext}"), recursive=True)
        assert len(self.image_paths) > 0, f"error2"
        self.pairs = []
        missing = []
        for img_path in sorted(self.image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(depth_root, stem + ".npy")
            if os.path.exists(depth_path):
                self.pairs.append((img_path, depth_path))
            else:
                missing.append(stem)
        if missing:
            print(f"error3")
        assert len(self.pairs) > 0, "error4"
        print(f"success: {len(self.pairs)}")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img_path, depth_path = self.pairs[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"error5")
        img_bgr, _pad_info = letterbox_resize(
            img_bgr, self.img_size, interpolation=cv2.INTER_LINEAR, pad_value=0
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth_raw = np.load(depth_path)
        depth_raw = np.squeeze(depth_raw).astype(np.float32)
        depth_raw, _pad_info_d = letterbox_resize(
            depth_raw, self.img_size, interpolation=cv2.INTER_NEAREST, pad_value=0.0
        )
        valid_mask = ((depth_raw > self.min_gt) & (depth_raw <= self.max_gt)).astype(np.float32)
        img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        depth_t = torch.from_numpy(depth_raw).float()
        mask_t = torch.from_numpy(valid_mask).float()
        return img_t, depth_t, mask_t
def finetune(
    image_root,
    depth_root,
    init_ckpt,
    depth_ckpt_path,
    save_dir="./checkpoints_stage2",
    epochs=20,
    batch_size=4,
    lr=1e-5,
    lambda_smooth=0.05,
    lambda_fidelity=1.0,
    min_gt=1e-6,
    max_gt=80.0,
    depth_encoder="vitb",
    max_steps=None,
):
    os.makedirs(save_dir, exist_ok=True)
    frbnet = FIINet(number_K=10, lamda=0.1)
    load_frbnet_flexible(frbnet, init_ckpt)
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
    dataset = RealNightGTDataset(image_root, depth_root, min_gt=min_gt, max_gt=max_gt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                         drop_last=True, pin_memory=True, persistent_workers=True)
    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("start(AMP)")
    for epoch in range(epochs):
        epoch_loss = 0.0
        step = -1
        for step, (real_img, gt_depth, valid_mask) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                print(f"success max_steps={max_steps}")
                break
            real_img = real_img.to(DEVICE)
            gt_depth = gt_depth.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                raw_enhanced = frbnet(real_img)
                enhanced01 = to_valid_image(raw_enhanced)
                enhanced_for_depth = normalize_for_depth_model(enhanced01, DEVICE)
                depth_pred = depth_model(enhanced_for_depth)
                loss_consist, aligned_pred = masked_depth_consistency_loss(depth_pred, gt_depth, valid_mask)
                loss_smooth = edge_aware_smoothness_loss(aligned_pred.unsqueeze(1), enhanced01)
                loss_fid = fidelity_loss(enhanced01, real_img, weight=lambda_fidelity)
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
        if actual_steps <= 0:
            print(f"error6")
            continue
        print(f"==> Epoch {epoch+1} aveloss: {epoch_loss/actual_steps:.4f}")
        ckpt_path = os.path.join(save_dir, f"frbnet_stage2v2_epoch{epoch+1}.pth")
        torch.save(frbnet.state_dict(), ckpt_path)
        print(f"save: {ckpt_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fittrain")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--init-ckpt", required=True)
    parser.add_argument("--depth-ckpt", required=True)
    parser.add_argument("--max-gt", type=float, default=80.0)
    parser.add_argument("--save-dir", default="./checkpoints_stage2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-fidelity", type=float, default=1.0)
    parser.add_argument("--depth-encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    finetune(
        image_root=args.image_root,
        depth_root=args.depth_root,
        init_ckpt=args.init_ckpt,
        depth_ckpt_path=args.depth_ckpt,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_smooth=args.lambda_smooth,
        lambda_fidelity=args.lambda_fidelity,
        depth_encoder=args.depth_encoder,
        max_steps=args.max_steps,
        max_gt=args.max_gt,
    )