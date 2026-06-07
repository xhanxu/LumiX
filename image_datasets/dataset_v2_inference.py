import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def image_resize(img, max_size=512, force_square=False):
    """
    Resize an image.
    Args:
        img: PIL image
        max_size: maximum output size
        force_square: if True, force square output; otherwise preserve aspect ratio
    """
    w, h = img.size
    
    if force_square:
        return img.resize((max_size, max_size))
    else:
        if w >= h:
            new_w = max_size
            new_h = int((max_size / w) * h)
        else:
            new_h = max_size
            new_w = int((max_size / h) * w)
        return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class HypersimDataset(Dataset):
    def __init__(self, data_root, modalities=['color', 'diffuse_illumination', 'lambertian', 'diffuse_reflectance', 'depth', 'normal'], 
                 captions_dir=None, img_size=512, random_ratio=False, force_square=True, condition_only=False, condition_modalities=None):
        """
        Hypersim dataset loader
        
        Args:
            data_root: root directory for processed images, e.g. /path/to/processed_images
            modalities: modalities to load, e.g. ['color', 'diffuse_illumination', 'lambertian', 'diffuse_reflectance']
            captions_dir: caption directory; defaults to data_root/captions_blip2 when None
            img_size: image size
            random_ratio: whether to use random crop aspect ratios
            force_square: whether to force square output; otherwise preserve aspect ratio
            condition_only: if True, only load condition modalities and captions
            condition_modalities: condition modalities; when omitted, the first modality is used
        """
        self.data_root = data_root
        self.modalities = modalities
        self.img_size = img_size
        self.random_ratio = random_ratio
        self.force_square = force_square
        self.condition_only = condition_only
        
        if condition_only:
            if condition_modalities is not None:
                self.actual_modalities = condition_modalities
            else:
                self.actual_modalities = [modalities[0]]
        else:
            self.actual_modalities = modalities

        if captions_dir is None:
            self.captions_dir = os.path.join(data_root, 'captions_blip2')
        else:
            self.captions_dir = captions_dir
        
        self._validate_modalities()
        
        self.sample_ids = self._get_sample_ids()
        
        self._print_dataset_info()
        
    def _validate_modalities(self):
        """Validate that modality folders exist."""
        available_modalities = []
        if os.path.exists(self.data_root):
            available_modalities = [d for d in os.listdir(self.data_root) 
                                  if os.path.isdir(os.path.join(self.data_root, d))]
        
        print(f"Available modalities: {available_modalities}")
        
        for modality in self.actual_modalities:
            modality_path = os.path.join(self.data_root, modality)
            if not os.path.exists(modality_path):
                raise ValueError(f"Modality directory does not exist: {modality_path}")
    
    def _get_sample_ids(self):
        """Collect sample IDs from the first required modality."""
        first_modality_dir = os.path.join(self.data_root, self.actual_modalities[0])
        
        files = [f for f in os.listdir(first_modality_dir) 
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        
        sample_ids = set()
        first_modality = self.actual_modalities[0]
        
        for file in files:
            base_name = os.path.splitext(file)[0]
            
            modality_suffix = f"_{first_modality}"
            if base_name.endswith(modality_suffix):
                sample_id = base_name[:-len(modality_suffix)]
                sample_ids.add(sample_id)
            else:
                sample_ids.add(base_name)
        
        valid_sample_ids = []
        for sample_id in sample_ids:
            if self._has_all_required_modalities(sample_id):
                valid_sample_ids.append(sample_id)
        
        valid_sample_ids.sort()
        return valid_sample_ids
        return valid_sample_ids
    
    def _has_all_required_modalities(self, sample_id):
        """Check that all required modality files exist."""
        for modality in self.actual_modalities:
            filename_standard = f"{sample_id}_{modality}.png"
            file_path_standard = os.path.join(self.data_root, modality, filename_standard)
            
            found = False
            if os.path.exists(file_path_standard):
                found = True
            else:
                for ext in ['.png', '.jpg', '.jpeg']:
                    filename_direct = f"{sample_id}{ext}"
                    file_path_direct = os.path.join(self.data_root, modality, filename_direct)
                    if os.path.exists(file_path_direct):
                        found = True
                        break
            
            if not found:
                return False
        return True
    
    def _load_image(self, sample_id, modality, crop_ratio=None, final_size=None):
        """
        Load an image for a modality.
        
        Args:
            sample_id: sample ID
            modality: modality name
            crop_ratio: preselected crop ratio; no crop when None
            final_size: preselected final size (width, height); defaults are used when None
        """
        filename_standard = f"{sample_id}_{modality}.png"
        file_path_standard = os.path.join(self.data_root, modality, filename_standard)
        
        file_path = None
        if os.path.exists(file_path_standard):
            file_path = file_path_standard
        else:
            for ext in ['.png', '.jpg', '.jpeg']:
                filename_direct = f"{sample_id}{ext}"
                file_path_direct = os.path.join(self.data_root, modality, filename_direct)
                if os.path.exists(file_path_direct):
                    file_path = file_path_direct
                    break
        
        if file_path is None:
            raise FileNotFoundError(f"File not found: {sample_id} in modality {modality}")
        
        img = Image.open(file_path).convert('RGB')
        
        if crop_ratio is not None and crop_ratio != "default":
            img = crop_to_aspect_ratio(img, crop_ratio)
        
        if final_size is not None:
            img = img.resize(final_size)
        else:
            img = image_resize(img, self.img_size, force_square=self.force_square)
            
            if not self.force_square:
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
            else:
                size = (self.img_size // 32) * 32
                img = img.resize((size, size))
        
        img_array = np.array(img) / 255.0
        
        img = torch.from_numpy((img_array * 2.0) - 1.0).float()
        img = img.permute(2, 0, 1)
        
        return img
    
    def _load_caption(self, sample_id):
        """Load a caption when a caption directory is provided."""
        if self.captions_dir is None:
            return f"interior scene from {sample_id}"
        
        caption_filename_standard = f"{sample_id}_color.json"
        caption_path_standard = os.path.join(self.captions_dir, caption_filename_standard)
        
        caption_filename_direct = f"{sample_id}.json"
        caption_path_direct = os.path.join(self.captions_dir, caption_filename_direct)
        
        caption_path = None
        if os.path.exists(caption_path_standard):
            caption_path = caption_path_standard
        elif os.path.exists(caption_path_direct):
            caption_path = caption_path_direct
        
        if caption_path:
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('caption', f"interior scene from {sample_id}")
            except Exception as e:
                print(f"Failed to load caption {caption_path}: {e}")
                return f"interior scene from {sample_id}"
        
        return f"interior scene from {sample_id}"
    
    def _print_dataset_info(self):
        """Print dataset information."""
        print(f"\n=== Hypersim Dataset Info ===")
        print(f"Root directory: {self.data_root}")
        print(f"Requested modalities: {self.modalities}")
        print(f"Loaded modalities: {self.actual_modalities}")
        print(f"Condition-only mode: {self.condition_only}")
        print(f"Number of samples: {len(self.sample_ids)}")
        print(f"Image size: {self.img_size}")
        print(f"Image mode: {'force square' if self.force_square else 'preserve aspect ratio'}")
        print(f"Caption directory: {self.captions_dir}")
        
        for modality in self.actual_modalities:
            modality_dir = os.path.join(self.data_root, modality)
            if os.path.exists(modality_dir):
                file_count = len([f for f in os.listdir(modality_dir) 
                                if f.endswith('.jpg') or f.endswith('.png')])
                print(f"  {modality}: {file_count} files")
    
    def get_available_modalities(self):
        """Return all available modalities."""
        if not os.path.exists(self.data_root):
            return []
        return [d for d in os.listdir(self.data_root) 
                if os.path.isdir(os.path.join(self.data_root, d))]
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        try:
            sample_id = self.sample_ids[idx]
            
            crop_ratio = None
            final_size = None
            
            if self.random_ratio:
                crop_ratio = random.choice(["16:9", "default", "1:1", "4:3"])
            else:
                crop_ratio = "default"
            
            first_modality = self.actual_modalities[0]
            
            temp_file_path = None
            temp_filename_standard = f"{sample_id}_{first_modality}.png"
            temp_file_path_standard = os.path.join(self.data_root, first_modality, temp_filename_standard)
            
            if os.path.exists(temp_file_path_standard):
                temp_file_path = temp_file_path_standard
            else:
                for ext in ['.png', '.jpg', '.jpeg']:
                    temp_filename_direct = f"{sample_id}{ext}"
                    temp_file_path_direct = os.path.join(self.data_root, first_modality, temp_filename_direct)
                    if os.path.exists(temp_file_path_direct):
                        temp_file_path = temp_file_path_direct
                        break
            
            if temp_file_path is None:
                raise FileNotFoundError(f"File not found: {sample_id} in modality {first_modality}")
                
            temp_img = Image.open(temp_file_path).convert('RGB')
            
            if crop_ratio != "default":
                temp_img = crop_to_aspect_ratio(temp_img, crop_ratio)
            
            temp_img = image_resize(temp_img, self.img_size, force_square=self.force_square)
            
            if not self.force_square:
                w, h = temp_img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                final_size = (new_w, new_h)
            else:
                size = (self.img_size // 32) * 32
                final_size = (size, size)
            
            images = {}
            for modality in self.actual_modalities:
                images[modality] = self._load_image(sample_id, modality, crop_ratio, final_size)
            
            caption = self._load_caption(sample_id)
            
            result = {
                'images': images,
                'caption': caption,
                'sample_id': sample_id
            }
            
            return result
            
        except Exception as e:
            print(f"Failed to load sample {idx} ({self.sample_ids[idx] if idx < len(self.sample_ids) else 'unknown'}): {e}")
            return self.__getitem__(random.randint(0, len(self.sample_ids) - 1))


def hypersim_loader(train_batch_size, num_workers, **args):
    """Hypersim dataset loader"""
    dataset = HypersimDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
