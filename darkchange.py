import argparse
import os
import random
import cv2
import numpy as np
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
class DarkISP:
    def __init__(
        self,
        darkness_range=(0.05, 0.15),
        gamma_range=(2.0, 3.5),
        rgb_range=(0.8, 0.1),
        red_range=(1.9, 2.4),
        blue_range=(1.5, 1.9),
        quantization=(4, 6, 8),
        noise_scale=1.0,
        color_cast_guard=0.6,
    ):
        self.darkness_low, self.darkness_high = darkness_range
        self.gamma_low, self.gamma_high = gamma_range
        self.color_cast_guard = color_cast_guard
        self.xyz2cams = [
            [[1.0234, -0.2969, -0.2266],
             [-0.5625, 1.6328, -0.0469],
             [-0.0703, 0.2188, 0.6406]],
            [[0.4913, -0.0541, -0.0202],
             [-0.613, 1.3513, 0.2906],
             [-0.1564, 0.2151, 0.7183]],
            [[0.838, -0.263, -0.0639],
             [-0.2887, 1.0725, 0.2496],
             [-0.0627, 0.1427, 0.5438]],
            [[0.6596, -0.2079, -0.0562],
             [-0.4782, 1.3016, 0.1933],
             [-0.097, 0.1581, 0.5181]],
        ]
        self.rgb2xyz = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
        self.rgb_mean, self.rgb_var = rgb_range
        self.red_low, self.red_high = red_range
        self.blue_low, self.blue_high = blue_range
        self.quantization = list(quantization)
        self.noise_scale = noise_scale
    def __call__(self, img_float01):
        img, meta = self.inverse_process(img_float01)
        img, meta2 = self.low_light(img)
        img = self.process(img, meta)
        img = np.clip(img, 0.0, 1.0)
        if self.color_cast_guard > 0:
            img = self._gray_world_correct(img, self.color_cast_guard)
        info = {**meta, **meta2}
        return img.astype(np.float32), info
    @staticmethod
    def _gray_world_correct(img, strength):
        means = img.reshape(-1, 3).mean(axis=0)
        overall = means.mean()
        gains = overall / np.maximum(means, 1e-6)
        gains = 1.0 + strength * (gains - 1.0)
        return np.clip(img * gains[np.newaxis, np.newaxis, :], 0.0, 1.0)
    def inverse_process(self, img):
        gamma = np.random.uniform(self.gamma_low, self.gamma_high)
        img = np.maximum(img, 1e-8) ** gamma
        xyz2cam = random.choice(self.xyz2cams)
        rgb2xyz = np.array(self.rgb2xyz)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
        img = self.apply_ccm(img, rgb2cam)
        img = np.clip(img, 0.0, None)
        rgb_gain = 1.0 / np.random.normal(self.rgb_mean, self.rgb_var)
        red_gain = np.random.uniform(self.red_low, self.red_high)
        blue_gain = np.random.uniform(self.blue_low, self.blue_high)
        gains = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
        gains = gains[np.newaxis, np.newaxis, :]
        img = img * gains
        meta = {
            'gamma': gamma,
            'rgb2cam': rgb2cam,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'rgb_gain': rgb_gain,
        }
        return img, meta
    def low_light(self, img):
        darkness = np.random.uniform(self.darkness_low, self.darkness_high)
        img = img * darkness
        shot_noise, read_noise = self.random_noise_levels()
        variance = img * shot_noise + read_noise
        variance = np.maximum(variance, 1e-8)
        noise = np.random.normal(0, np.sqrt(variance) * self.noise_scale, size=np.shape(img))
        img = img + noise
        bits = random.choice(self.quantization)
        img_quan1 = img * 255.0
        img_quan_bit = img_quan1 / bits
        img_quan2 = np.around(img_quan_bit) * bits
        img = img_quan2 / 255.0
        meta2 = {
            'darkness': darkness,
            'quan_bits': bits,
            'noise_variance': float(np.mean(variance)),
        }
        return img, meta2
    def process(self, img, meta):
        gamma = meta['gamma']
        rgb2cam = meta['rgb2cam']
        red_gain = meta['red_gain']
        blue_gain = meta['blue_gain']
        green_gain = np.ones_like(red_gain)
        gains = np.stack([red_gain, green_gain, blue_gain], axis=-1)
        gains = gains[np.newaxis, np.newaxis, :]
        img = img * gains
        cam2rgb = np.linalg.inv(rgb2cam)
        img = self.apply_ccm(img, cam2rgb)
        img = np.clip(img, 0.0, None)
        img = np.maximum(img, 1e-8) ** (1.0 / gamma)
        return img
    @staticmethod
    def apply_ccm(image, ccm):
        shape = image.shape
        image = np.reshape(image, [-1, 3])
        image = np.tensordot(image, ccm, axes=[[-1], [-1]])
        return np.reshape(image, shape)
    @staticmethod
    def random_noise_levels():
        log_min_shot_noise = np.log(0.0001)
        log_max_shot_noise = np.log(0.012)
        log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
        shot_noise = np.exp(log_shot_noise)
        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
        read_noise = np.exp(log_read_noise)
        return shot_noise, read_noise
def load_image_as_float01(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0
def save_float01_as_image(img_float01, path):
    img_uint8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
def process_one(dark_isp, in_path, out_path, seed=None, verbose=True):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    img = load_image_as_float01(in_path)
    night_img, info = dark_isp(img)
    save_float01_as_image(night_img, out_path)
    if verbose:
        print(f'  -> {out_path}  '
              f'(darkness={info["darkness"]:.4f}, gamma={info["gamma"]:.2f}, '
              f'quan_bits={info["quan_bits"]}, noise_var={info["noise_variance"]:.5f})')
def main():
    parser = argparse.ArgumentParser(description='Dark-ISP')
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--darkness', nargs=2, type=float, default=(0.05, 0.15))
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--gamma', nargs=2, type=float, default=(2.0, 3.5))
    parser.add_argument('--fix-cast', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    dark_isp = DarkISP(
        darkness_range=tuple(args.darkness),
        gamma_range=tuple(args.gamma),
        noise_scale=args.noise_scale,
        color_cast_guard=args.fix_cast,
    )
    batch_mode = args.input_dir is not None or args.output_dir is not None
    if batch_mode:
        os.makedirs(args.output_dir, exist_ok=True)
        filenames = sorted(
            f for f in os.listdir(args.input_dir)
            if f.lower().endswith(IMG_EXTS)
        )
        if not filenames:
            print(f'not found')
            return
        print(f'find {len(filenames)} photos')
        for idx, fname in enumerate(filenames):
            in_path = os.path.join(args.input_dir, fname)
            name, _ = os.path.splitext(fname)
            out_path = os.path.join(args.output_dir, f'{name}_night.jpg')
            per_image_seed = None
            if args.seed is not None:
                per_image_seed = args.seed + idx
            try:
                process_one(dark_isp, in_path, out_path, seed=per_image_seed)
            except Exception as e:
                print(f' fail: {e}')
        print(f'save in {args.output_dir}')
    else:
        if not args.input or not args.output:
            parser.error('error2')
        process_one(dark_isp, args.input, args.output, seed=args.seed)
        print(f'save in: {args.output}')
if __name__ == '__main__':
    main()