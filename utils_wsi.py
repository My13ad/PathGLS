import openslide
import numpy as np
import cv2
from PIL import Image
import random
import os

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("[Warning] 'tifffile' library not found. Generic TIFF files may fail to load.")


class TiffBackendWrapper:
    def __init__(self, wsi_path):
        self.tif = tifffile.TiffFile(wsi_path)
        self.page_ref = None

        if len(self.tif.series) > 0:
            self.page_ref = self.tif.series[0]
        elif len(self.tif.pages) > 0:
            self.page_ref = self.tif.pages[0]
        else:
            raise ValueError("TIFF file contains no readable pages/series (File likely corrupted).")

        shape = self.page_ref.shape
        if len(shape) == 3:
            self.h, self.w, self.c = shape
        elif len(shape) == 2:
            self.h, self.w = shape
            self.c = 1
        elif len(shape) == 4:
            self.h, self.w = shape[-3], shape[-2]
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")

        self.dimensions = (self.w, self.h)
        self.level_count = 1
        self.properties = {
            openslide.PROPERTY_NAME_MPP_X: 0.25,
            openslide.PROPERTY_NAME_MPP_Y: 0.25,
        }

    def read_region(self, location, level, size):
        x, y = location
        target_w, target_h = size
        try:
            img_data = self.page_ref.asarray(out='memmap')
            if img_data.ndim == 2:
                region = img_data[y : y + target_h, x : x + target_w]
                region = np.stack((region, region, region), axis=-1)
            elif img_data.ndim == 3:
                region = img_data[y : y + target_h, x : x + target_w, :]
            else:
                region = img_data[..., y : y + target_h, x : x + target_w, :]
                region = np.squeeze(region)
                if region.ndim == 2:
                    region = np.stack((region, region, region), axis=-1)

            if region.shape[-1] > 3:
                region = region[..., :3]

            if region.dtype != np.uint8:
                if region.dtype == np.uint16:
                    region = (region / 256).astype(np.uint8)
                else:
                    region = region.astype(np.uint8)

            return Image.fromarray(region)
        except Exception:
            return Image.new("RGB", size, (0, 0, 0))

    def close(self):
        self.tif.close()


class WSIProcessor:
    def __init__(self, wsi_path, target_mag=40, patch_size=512):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.target_mag = target_mag
        self.slide = None
        self.is_tifffile = False

        try:
            self.slide = openslide.OpenSlide(wsi_path)
        except Exception as e_os:
            if TIFFFILE_AVAILABLE and wsi_path.lower().endswith((".tiff", ".tif")):
                try:
                    self.slide = TiffBackendWrapper(wsi_path)
                    self.is_tifffile = True
                except Exception as e_tif:
                    raise RuntimeError(f"WSI Load Failed. OpenSlide: {e_os}; Tifffile: {e_tif}")
            else:
                raise e_os

        try:
            self.mpp = float(
                self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25)
            )
        except Exception:
            self.mpp = 0.25
        self.level = 0

    def get_tissue_mask(self, downsample_factor=64):
        w, h = self.slide.dimensions
        target_w = w // downsample_factor
        target_h = h // downsample_factor
        try:
            if hasattr(self.slide, "get_thumbnail") and not self.is_tifffile:
                img_thumb = self.slide.get_thumbnail((target_w, target_h))
            else:
                stride = downsample_factor
                if self.slide.is_tifffile:
                    full = self.slide.page_ref.asarray(out='memmap')
                    thumb = full[::stride, ::stride]
                    if thumb.ndim == 2:
                        thumb = np.stack((thumb,) * 3, axis=-1)
                    elif thumb.ndim == 3 and thumb.shape[2] > 3:
                        thumb = thumb[..., :3]
                    if thumb.dtype == np.uint16:
                        thumb = (thumb / 256).astype(np.uint8)
                    img_thumb = Image.fromarray(thumb.astype(np.uint8))
                else:
                    img_thumb = Image.new("RGB", (target_w, target_h))
        except Exception:
            return None, 1.0

        img_thumb_np = np.array(img_thumb)
        if len(img_thumb_np.shape) == 3:
            gray = cv2.cvtColor(img_thumb_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_thumb_np

        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        real_scale = w / img_thumb.size[0]
        return mask, real_scale

    def sample_patches(self, num_patches=15):
        mask, scale = self.get_tissue_mask(downsample_factor=64)
        if mask is None:
            return []

        y_idxs, x_idxs = np.where(mask > 0)
        if len(x_idxs) == 0:
            return []

        coords = list(zip(x_idxs, y_idxs))
        if len(coords) > num_patches:
            sampled_coords = random.sample(coords, num_patches)
        else:
            sampled_coords = coords

        patches_data = []
        for x_thumb, y_thumb in sampled_coords:
            x_l0 = int(x_thumb * scale)
            y_l0 = int(y_thumb * scale)

            offset = int(scale // 2)
            x_l0 += random.randint(-offset, offset)
            y_l0 += random.randint(-offset, offset)

            w, h = self.slide.dimensions
            x_l0 = max(0, min(x_l0, w - self.patch_size))
            y_l0 = max(0, min(y_l0, h - self.patch_size))

            try:
                patch_img = self.slide.read_region(
                    (x_l0, y_l0), self.level, (self.patch_size, self.patch_size)
                )
                patch_img = patch_img.convert("RGB")
                patches_data.append({"patch_img": patch_img, "coords": (x_l0, y_l0)})
            except Exception:
                continue

        return patches_data
