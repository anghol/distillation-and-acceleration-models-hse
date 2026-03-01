import os
import time
import shutil

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.anime_gan import GeneratorV1
from utils.common import load_checkpoint, RELEASED_WEIGHTS
from utils.image_processing import resize_image, normalize_input, denormalize_input
from utils import read_image, is_image_file, color_transfer_pytorch
from tqdm import tqdm


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - started_at
        print(f"Processed in {elapsed:.3f}s")
        return result
    return wrap


def auto_load_weight(weight, version=None, map_location=None):
    """Auto load Generator version from weight."""
    weight_name = os.path.basename(weight).lower()
    if version is not None:
        version = version.lower()
        assert version in {"v1", "v2", "v3"}, f"Version {version} does not exist"
        # If version is provided, use it.
        cls = {
            "v1": GeneratorV1,
            # "v2": GeneratorV2,
            # "v3": GeneratorV3
        }[version]
    else:
        # Try to get class by name of weight file    
        # For convenenice, weight should start with classname
        # e.g: Generatorv2_{anything}.pt
        if weight_name in RELEASED_WEIGHTS:
            version = RELEASED_WEIGHTS[weight_name][0]
            return auto_load_weight(weight, version=version, map_location=map_location)

        # elif weight_name.startswith("generatorv2"):
        #     cls = GeneratorV2
        # elif weight_name.startswith("generatorv3"):
        #     cls = GeneratorV3
        elif weight_name.startswith("generator"):
            cls = GeneratorV1
        else:
            raise ValueError((f"Can not get Model from {weight_name}, "
                               "you might need to explicitly specify version"))
    model = cls()
    load_checkpoint(model, weight, strip_optimizer=True, map_location=map_location)
    model.eval()
    return model


class Predictor:
    """
    Generic class for transfering Image to anime like image.
    """
    def __init__(
        self,
        weight='hayao',
        device='cuda',
        amp=True,
        retain_color=False,
        imgsz=None,
    ):
        if not torch.cuda.is_available():
            device = 'cpu'
            # Amp not working on cpu
            amp = False
            print("Use CPU device")
        else:
            print(f"Use GPU {torch.cuda.get_device_name()}")
        
        self.imgsz = imgsz
        self.retain_color = retain_color
        self.amp = amp  # Automatic Mixed Precision
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.device = torch.device(device)
        self.G = auto_load_weight(weight, map_location=device)
        self.G.to(self.device)

    def transform_and_show(
        self,
        image_path,
        figsize=(18, 10),
        save_path=None
    ):
        # image = resize_image(read_image(image_path))
        image = self.read_and_resize(image_path)
        anime_img = self.transform(image)
        anime_img = anime_img.astype('uint8')

        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1, 2, 1)
        # plt.title("Input")
        plt.imshow(image)
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        # plt.title("Anime style")
        plt.imshow(anime_img[0])
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def transform(self, image, denorm=True):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        with torch.no_grad():
            image = self.preprocess_images(image)
            fake = self.G(image)

            # Transfer color of fake image look similiar color as image
            if self.retain_color:
                fake = color_transfer_pytorch(fake, image)
                fake = (fake / 0.5) - 1.0  # remap to [-1. 1]
            fake = fake.detach().cpu().numpy()
            
            # Channel last
            fake = fake.transpose(0, 2, 3, 1)

            if denorm:
                fake = denormalize_input(fake, dtype=np.uint8)
            return fake

    def read_and_resize(self, path, max_size=1536):
        image = read_image(path)
        _, ext = os.path.splitext(path)
        h, w = image.shape[:2]

        if self.imgsz is not None:
            image = resize_image(image, width=self.imgsz)
        elif max(h, w) > max_size:
            print(f"Image {os.path.basename(path)} is too big ({h}x{w}), resize to max size {max_size}")
            image = resize_image(
                image,
                width=max_size if w > h else None,
                height=max_size if w < h else None,
            )
            cv2.imwrite(path.replace(ext, ".jpg"), image[:,:,::-1])
        else:
            image = resize_image(image)

        return image

    @profile
    def transform_file(self, file_path, save_path):
        if not is_image_file(save_path):
            raise ValueError(f"{save_path} is not valid")

        image = self.read_and_resize(file_path)
        anime_img = self.transform(image)[0]
        cv2.imwrite(save_path, anime_img[..., ::-1])
        print(f"Anime image saved to {save_path}")

    @profile
    def transform_gif(self, file_path, save_path, batch_size=4):
        import imageio

        def _preprocess_gif(img):
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return resize_image(img)

        images = imageio.mimread(file_path)
        images = np.stack([
            _preprocess_gif(img)
            for img in images
        ])

        print(images.shape)

        anime_gif = np.zeros_like(images)

        for i in tqdm(range(0, len(images), batch_size)):
            end = i + batch_size
            anime_gif[i: end] = self.transform(
                images[i: end]
            )

        if end < len(images) - 1:
            # transform last frame
            print("LAST", images[end: ].shape)
            anime_gif[end:] = self.transform(images[end:])

        print(anime_gif.shape)
        imageio.mimsave(
            save_path,
            anime_gif,
            
        )
        print(f"Anime image saved to {save_path}")

    @profile
    def transform_in_dir(self, img_dir, dest_dir, max_images=0, img_size=(512, 512)):
        '''
        Read all images from img_dir, transform and write the result
        to dest_dir

        '''
        os.makedirs(dest_dir, exist_ok=True)

        files = os.listdir(img_dir)
        files = [f for f in files if is_image_file(f)]
        print(f'Found {len(files)} images in {img_dir}')

        if max_images:
            files = files[:max_images]

        bar = tqdm(files)
        for fname in bar:
            path = os.path.join(img_dir, fname)
            image = self.read_and_resize(path)
            anime_img = self.transform(image)[0]
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            cv2.imwrite(os.path.join(dest_dir, f'{fname}.jpg'), anime_img[..., ::-1])
            bar.set_description(f"{fname} {image.shape}")

    def preprocess_images(self, images):
        '''
        Preprocess image for inference

        @Arguments:
            - images: np.ndarray

        @Returns
            - images: torch.tensor
        '''
        images = images.astype(np.float32)

        # Normalize to [-1, 1]
        images = normalize_input(images)
        images = torch.from_numpy(images)

        images = images.to(self.device)

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # channel first
        images = images.permute(0, 3, 1, 2)

        return images


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight',
        type=str,
        default="hayao:v1",
        help=f'Model weight, can be path or pretrained {tuple(RELEASED_WEIGHTS.keys())}'
    )
    parser.add_argument('--src', type=str, help='Source, can be directory contains images, image file or video file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device, cuda or cpu')
    parser.add_argument('--imgsz', type=int, default=None, help='Resize image to specified size if provided')
    parser.add_argument('--out', type=str, default='inference_images', help='Output, can be directory or file')
    parser.add_argument(
        '--retain-color',
        action='store_true',
        help='If provided the generated image will retain original color of input image')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    predictor = Predictor(
        args.weight,
        args.device,
        retain_color=args.retain_color,
        imgsz=args.imgsz,
    )

    if not os.path.exists(args.src):
        raise FileNotFoundError(args.src)

    if os.path.isdir(args.src):
        predictor.transform_in_dir(args.src, args.out)
    elif os.path.isfile(args.src):
        save_path = args.out
        if not is_image_file(args.out):
            os.makedirs(args.out, exist_ok=True)
            save_path = os.path.join(args.out, os.path.basename(args.src))

        if args.src.endswith('.gif'):
            # GIF file
            predictor.transform_gif(args.src, save_path, args.batch_size)
        else:
            predictor.transform_file(args.src, save_path)
    else:
        raise NotImplementedError(f"{args.src} is not supported")
