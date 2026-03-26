import os
import random
import cv2
import numpy as np
from glob import glob
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset

from utils import normalize_input, fast_numpyio, read_image, resize_image


CACHE_DIR = '.tmp'


def get_random_crop(image, crop_height, crop_width):

    max_x = max(image.shape[1] - crop_width, 0)
    max_y = max(image.shape[0] - crop_height, 0)

    x = np.random.randint(0, max_x) if max_x != 0 else 0
    y = np.random.randint(0, max_y) if max_y != 0 else 0

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

    
class AnimeDataSet(Dataset):
    def __init__(
        self,
        anime_image_dir,
        real_image_dir,
        debug_samples=0,
        cache=False,
        transform=None,
        imgsz=256,
        resize_method="resize"
    ):
        """   
        folder structure:
        - {anime_image_dir}  # E.g Hayao
            smooth
                1.jpg, ..., n.jpg
            style
                1.jpg, ..., n.jpg
        """
        self.cache = False  # Disable cache forever

        if isinstance(imgsz, list):
            # Get first imgsz
            imgsz = imgsz[0]

        self.debug_samples = debug_samples
        self.resize_method = resize_method
        self.image_files =  {}
        self.photo = 'train_photo'
        self.style = 'style'
        self.smooth = 'smooth'
        self.cache_files = {}
        self.anime_dirname = os.path.basename(anime_image_dir)
        self.imgsz = imgsz
        for dir, opt in [
            (real_image_dir, self.photo),
            (os.path.join(anime_image_dir, self.style), self.style),
            (os.path.join(anime_image_dir, self.smooth), self.smooth)
        ]:
            self.image_files[opt] = sorted(glob(os.path.join(dir, "*.*")))
            self.cache_files[opt] = [False] * len(self.image_files[opt])

        self.transform = transform
        self.cache_data()

        print(f'Dataset: real {self.len_photo} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or self.len_anime

    @property
    def len_photo(self):
        return len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        photo_idx = random.randint(0, self.len_photo - 1)
        anm_idx = index

        image = self.load_photo(photo_idx)
        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return {
            "image": torch.from_numpy(image),
            "anime": torch.from_numpy(anime),
            "anime_gray": torch.from_numpy(anime_gray),
            "smooth_gray": torch.from_numpy(smooth_gray)
        }

    def set_imgsz(self, imgsz):
        self.imgsz = imgsz

    def cache_data(self):
        if not self.cache:
            return
        
        cache_dir = os.path.join(CACHE_DIR, self.anime_dirname)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Caching image to npy for faster dataloader
        print("Caching data..")
        cache_nbytes = 0
        
        for opt, image_files in self.image_files.items():
            cache_sub_sir = os.path.join(cache_dir, opt)
            os.makedirs(cache_sub_sir, exist_ok=True)
            for index, img_file in enumerate(tqdm(image_files)):
                save_path = os.path.join(cache_sub_sir, f"{index}.npy")
                if os.path.exists(save_path):
                    continue  # Cache exist
                if opt == self.photo:
                    image = self.load_photo(index)
                    cache_nbytes += image.nbytes
                    fast_numpyio.save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.smooth:
                    cache_nbytes += image.nbytes
                    image = self.load_anime_smooth(index)
                    np.save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.style:
                    image, image_gray = self.load_anime(index)
                    cache_nbytes = cache_nbytes + image.nbytes + image_gray.nbytes
                    fast_numpyio.save(save_path, image)
                    save_path_gray = os.path.join(cache_sub_sir, f"{index}_gray.npy")
                    fast_numpyio.save(save_path_gray, image_gray)
                    self.cache_files[opt][index] = (save_path, save_path_gray)
                else:
                    raise ValueError(opt)
                    
        print(f"Cache saved to {cache_dir}, size={cache_nbytes/1e9} Gb")

    def load_photo(self, index) -> np.ndarray:
        if self.cache_files[self.photo][index]:
            fpath = self.cache_files[self.photo][index]
            image = fast_numpyio.load(fpath)
        else:
            fpath = self.image_files[self.photo][index]
            image = cv2.imread(fpath)[:,:,::-1]
            
            if self.resize_method == "resize":
                image = cv2.resize(image, (self.imgsz, self.imgsz))
            else:
                # Random Crop
                random_size = random.randint(
                    int(self.imgsz * 0.5),
                    int(self.imgsz * 1)
                )
                image = get_random_crop(
                    image, random_size, random_size)
                image = cv2.resize(image, (self.imgsz, self.imgsz))

            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            
        return image

    def load_anime(self, index) -> np.ndarray:
        if self.cache_files[self.style][index]:
            fpath, fpath_gray = self.cache_files[self.style][index]
            image = fast_numpyio.load(fpath)
            image_gray = fast_numpyio.load(fpath_gray)
        else:
            fpath = self.image_files[self.style][index]
            image = cv2.imread(fpath)[:,:,::-1]
            image = cv2.resize(image, (self.imgsz, self.imgsz))

            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
            image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)

            image_gray = self._transform(image_gray, addmean=False)
            image_gray = image_gray.transpose(2, 0, 1)
            image_gray = np.ascontiguousarray(image_gray)

            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)

        return image, image_gray

    def load_anime_smooth(self, index) -> np.ndarray:
        if self.cache_files[self.smooth][index]:
            fpath = self.cache_files[self.smooth][index]
            image = fast_numpyio.load(fpath)
        else:
            fpath = self.image_files[self.smooth][index]
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.imgsz, self.imgsz))
            image = np.stack([image, image, image], axis=-1)
            
            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            
        return image

    def _transform(self, img, addmean=False):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)


class TrainPhotosDataset(Dataset):
    def __init__(
        self,
        real_image_dir,
        transform=None,
        cache=False,
        imgsz=256,
        resize_method="resize"
    ):
        if isinstance(imgsz, list):
            imgsz = imgsz[0]  # Get first imgsz

        self.cache = cache
        self.imgsz = imgsz
        self.resize_method = resize_method
        self.transform = transform
        
        self.image_files =  sorted(glob(os.path.join(real_image_dir, "*.*")))
        self.cache_files = [False] * len(self.image_files)

        self.cache_data()

        print(f'Train dataset: {self.len_photo} photos')

    def __len__(self):
        return self.len_photo

    @property
    def len_photo(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.load_photo(index)
        return torch.from_numpy(image)

    def set_imgsz(self, imgsz):
        self.imgsz = imgsz

    def cache_data(self):
        if not self.cache:
            return
        
        cache_dir = '.tmp_train'
        os.makedirs(cache_dir, exist_ok=True)
        
        # Caching image to npy for faster dataloader
        print("Caching data..")
        cache_nbytes = 0

        for index, img_file in enumerate(tqdm(self.image_files)):
            save_path = os.path.join(cache_dir, f"{index}.npy")
            if os.path.exists(save_path):
                continue  # Cache exist.
            
            image = self.load_photo(index)
            cache_nbytes += image.nbytes
            fast_numpyio.save(save_path, image)
            self.cache_files[index] = save_path
            
        print(f"Cache saved to {cache_dir}, size={cache_nbytes/1e9} Gb")

    def load_photo(self, index) -> np.ndarray:
        if self.cache_files[index]:
            fpath = self.cache_files[index]
            image = fast_numpyio.load(fpath)
        else:
            fpath = self.image_files[index]
            image = cv2.imread(fpath)[:,:,::-1]
            
            if self.resize_method == "resize":
                image = cv2.resize(image, (self.imgsz, self.imgsz))
            else:
                # Random Crop
                random_size = random.randint(
                    int(self.imgsz * 0.5),
                    int(self.imgsz * 1)
                )
                image = get_random_crop(
                    image, random_size, random_size)
                image = cv2.resize(image, (self.imgsz, self.imgsz))

            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
        return image

    def _transform(self, img, addmean=False):
        if self.transform is not None:
            img =  self.transform(image=img)

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)


class TestPhotosDataset(Dataset):
    def __init__(
        self,
        image_dir,
        imgsz=None,
    ):
        if isinstance(imgsz, list) or isinstance(imgsz, tuple):
            imgsz = imgsz[0]

        self.imgsz = imgsz

        self.image_files = sorted(glob(os.path.join(image_dir, "*.*")))
        self.image_files = [f for f in self.image_files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

        print(f'Test dataset: {self.len_photo} photos')

    def __len__(self):
        return self.len_photo

    @property
    def len_photo(self):
        return len(self.image_files)

    def __getitem__(self, index):
        path = self.image_files[index]
        image = self.read_and_resize(path)
        image = self.preprocess_image(image)
        return image

    def read_and_resize(self, path, max_size=1024):
        image = read_image(path)
        h, w = image.shape[:2]

        if self.imgsz is not None:
            image = resize_image(image, width=self.imgsz)
        elif max(h, w) > max_size:
            image = resize_image(
                image,
                width=max_size if w > h else None,
                height=max_size if w < h else None,
            )
        else:
            image = resize_image(image)

        return image

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = normalize_input(image)  # to [-1, 1]
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image