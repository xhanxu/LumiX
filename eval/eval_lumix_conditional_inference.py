#!/usr/bin/env python3
"""
LumiX multi-modal conditional inference
Conditional inference script that supports using some modalities as conditions
and generates others.
"""

import argparse
import yaml
import os
import sys
import torch
import numpy as np
from PIL import Image
import json
import re
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LumiX processors for Tensor LoRA architecture
from src.flux.modules.layers_lumix import (
    LumiXSingleStreamProcessor,
    LumiXDoubleStreamProcessor,
)
from src.flux.util import load_checkpoint
from src.flux.util import load_ae, load_clip, load_flow_model2, load_t5

# Import sampling functions (same as training validation)
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from einops import rearrange

# Import dataset - use the modified inference version with condition support
from image_datasets.dataset_v2_inference import HypersimDataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_lumix_model(config):
    """Setup LumiX multi-modal model with Tensor LoRA processors"""
    device = config.get('device', 'cuda')
    model_type = config.get('model_type', 'flux-dev')
    offload = config.get('offload', False)
    
    is_schnell = model_type == "flux-schnell"
    
    # Load models
    print(f"Loading models on device: {device}")
    print(f"Offload enabled: {offload}")
    
    clip = load_clip(device)
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    ae = load_ae(model_type, device="cpu" if offload else device)
    model = load_flow_model2(model_type, device="cpu" if offload else device)
    
    # Ensure all models are on the correct device if not offloading
    if not offload:
        print(f"Moving all models to device: {device}")
        clip = clip.to(device)
        t5 = t5.to(device) 
        ae = ae.to(device)
        model = model.to(device)
    
    # Determine weight dtype based on mixed precision setting (match training)
    mixed_precision = config.get('mixed_precision', 'bf16')  # Default to bf16 as in training
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    print(f"Using weight dtype: {weight_dtype} (mixed_precision: {mixed_precision})")
    
    # Set up LumiX processors
    lora_local_path = config.get('lora_local_path')
    if lora_local_path and os.path.exists(lora_local_path):
        print(f"Loading LumiX multi-modal LoRA from: {lora_local_path}")
        
        lora_attn_procs = {}
        
        # LumiX parameters from config
        rank = config.get('rank', 8)  # Main rank
        rank_inter = config.get('rank_inter', 4)  # Intermediate rank
        network_alpha = config.get('network_alpha', None)
        lora_weight = config.get('lora_weight', 1.0)
        num_modalities = config.get('num_modalities', 5)
        
        print(f"LumiX Configuration: rank={rank}, rank_inter={rank_inter}, network_alpha={network_alpha}, lora_weight={lora_weight}, num_modalities={num_modalities}")
        
        # Determine which blocks to apply LoRA to
        double_blocks = config.get('double_blocks')
        single_blocks = config.get('single_blocks')
        
        if double_blocks is None:
            double_blocks_idx = list(range(19))
        else:
            double_blocks_idx = double_blocks if isinstance(double_blocks, list) else [int(idx) for idx in str(double_blocks).split(",")]

        if single_blocks is None:
            single_blocks_idx = list(range(38))
        else:
            single_blocks_idx = single_blocks if isinstance(single_blocks, list) else [int(idx) for idx in str(single_blocks).split(",")]

        # Set up LumiX processors
        print("Setting up LumiX processors...")
        
        for name, attn_processor in model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                print(f"Setting LumiX DoubleStream Processor for {name}")
                lora_attn_procs[name] = LumiXDoubleStreamProcessor(
                    dim=3072,
                    rank=rank,
                    network_alpha=network_alpha,
                    lora_weight=lora_weight,
                    num_modalities=num_modalities,
                )
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                print(f"Setting LumiX SingleStream Processor for {name}")
                lora_attn_procs[name] = LumiXSingleStreamProcessor(
                    dim=3072,
                    rank=rank,
                    network_alpha=network_alpha,
                    lora_weight=lora_weight,
                    num_modalities=num_modalities,
                )
            else:
                lora_attn_procs[name] = attn_processor

        model.set_attn_processor(lora_attn_procs)
        
        # Load LoRA weights
        print("Loading checkpoint weights...")
        checkpoint = load_checkpoint(lora_local_path, None, None)
        
        # Load weights into model
        model_state_dict = model.state_dict()
        model_state_dict.update(checkpoint)
        model.load_state_dict(model_state_dict, strict=False)
        print(f"✓ Loaded LoRA weights with {len(checkpoint)} parameters")
        
        # Move LoRA processors to correct device if not offloading
        if not offload:
            print(f"Moving LumiX processors to device: {device}")
            for name, processor in model.attn_processors.items():
                if hasattr(processor, 'to'):
                    processor.to(device)
    else:
        print(f"Warning: LoRA path {lora_local_path} does not exist or not specified")
    
    # Set requires_grad to False for inference
    clip.requires_grad_(False)
    t5.requires_grad_(False)
    ae.requires_grad_(False)
    model.requires_grad_(False)
    
    # Set model to eval mode for inference.
    # This disables dropout and batch norm updates
    model.eval()
    ae.eval()
    clip.eval()
    t5.eval()
    
    # Set model to bf16 to match training
    model = model.to(torch.bfloat16)
    
    return clip, t5, ae, model


def process_image_name(image_name):
    """
    Normalize an image name by removing modality suffixes.
    """
    suffixes_to_remove = [
        '_diffuse_reflectance', '_diffuse_illumination', '_color', 
        '_depth', '_normal', '_albedo', '_lambertian', '_tonemap'
    ]
    
    processed_name = image_name
    for suffix in suffixes_to_remove:
        if processed_name.endswith(suffix):
            processed_name = processed_name[:-len(suffix)]
            break
    
    return processed_name


def create_combined_image(images_dict, output_path, modality_order=None):
    """Create a combined grid image with all modalities"""
    if modality_order is None:
        modality_order = ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal']
    
    valid_images = []
    valid_names = []
    
    for modality in modality_order:
        # Check both the direct key and the 'modalities' sub-dict
        img = None
        if modality in images_dict and images_dict[modality] is not None:
            img = images_dict[modality]
        elif 'modalities' in images_dict and modality in images_dict['modalities']:
            img = images_dict['modalities'][modality]
        
        if img is not None:
            valid_images.append(img)
            valid_names.append(modality)
    
    if not valid_images:
        print("No valid images to combine")
        return
    
    # Assume all images have the same size
    img_width, img_height = valid_images[0].size
    
    # Create combined image (horizontal layout)
    combined_width = img_width * len(valid_images)
    combined_height = img_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    for i, img in enumerate(valid_images):
        combined_img.paste(img, (i * img_width, 0))
    
    combined_img.save(output_path)
    print(f"Combined image saved to: {output_path} (modalities: {', '.join(valid_names)})")


def generate_multimodal_images(model, vae, t5, clip, device, prompt, width=512, height=512,
                               num_steps=50, guidance=4.0, seed=42, true_gs=3.5, num_modalities=5,
                               neg_prompt=""):
    """
    Generate multi-modal images using the same approach as training validation
    Returns dict with all modalities: diffuse_reflectance, diffuse_illumination, color, depth, normal
    
    🔥 Key: Uses query broadcast attention mechanism (Stage 2) for all modalities
    - All modalities share the fine-tuned color query (idx=2)
    - Same noise for all modalities ensures alignment
    - Same text conditioning for all modalities
    
    Args:
        neg_prompt: Negative prompt to guide away from unwanted features (e.g., "blurry, low quality")
    """
    # Handle random seed
    current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
    
    # Use inference_mode for pure inference
    with torch.inference_mode():
        # Generate noise - same noise for all modalities (consistent with training)
        single_noise = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=current_seed)
        x = single_noise.repeat(num_modalities, 1, 1, 1)  # Repeat for all modalities
        
        # Get timesteps
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        
        torch.manual_seed(current_seed)
        
        # Prepare conditioning - replicate prompt for all modalities
        prompts = [prompt] * num_modalities
        # Use provided negative prompt or empty string
        neg_prompts = [neg_prompt if neg_prompt else ""] * num_modalities
        
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompts)
        neg_inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=neg_prompts)
        
        # Denoise - use the LumiX model directly
        x = denoise(
            model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
            timestep_to_start_cfg=0,
            neg_txt=neg_inp_cond['txt'],
            neg_txt_ids=neg_inp_cond['txt_ids'],
            neg_vec=neg_inp_cond['vec'],
            true_gs=true_gs,
            image_proj=None,
            neg_image_proj=None,
            ip_scale=1.0,
            neg_ip_scale=1.0,
        )
        
        # Decode - use VAE to decode latents
        x = unpack(x.float(), height, width)
        x = vae.decode(x)
        x = x.clamp(-1, 1)
        
        # Convert to PIL images - immediately move to CPU to save GPU memory
        output_imgs = []
        modality_names = ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal']
        
        for i in range(min(num_modalities, len(modality_names))):
            x_modal = rearrange(x[i], "c h w -> h w c").cpu()  # Move to CPU immediately
            output_img = Image.fromarray((127.5 * (x_modal + 1.0)).byte().numpy())
            output_imgs.append(output_img)
            del x_modal  # Clean up intermediate tensors
        
        del x  # Clean up decoded tensor
    
    # Return results as dict
    results = {
        'diffuse_reflectance': output_imgs[0] if len(output_imgs) > 0 else None,
        'diffuse_illumination': output_imgs[1] if len(output_imgs) > 1 else None,
        'color': output_imgs[2] if len(output_imgs) > 2 else None,
        'depth': output_imgs[3] if len(output_imgs) > 3 else None,
        'normal': output_imgs[4] if len(output_imgs) > 4 else None,
        'all': output_imgs
    }
    
    return results


def generate_conditional_multimodal_images(model, vae, t5, clip, device, prompt, 
                                         condition_images, condition_modalities, target_modalities,
                                         all_modalities, width=512, height=512, num_steps=50, 
                                         guidance=4.0, seed=42, true_gs=3.5, neg_prompt=""):
    """
    Generate multi-modal images with some modalities as conditions
    
    Args:
        model: The trained flux model
        vae: VAE for encoding/decoding
        t5, clip: Text encoders
        device: Device to run on
        prompt: Text prompt for generation
        condition_images: List of PIL Images for condition modalities (ordered by condition_modalities)
        condition_modalities: List of modality names that are conditions
        target_modalities: List of modality names to generate
        all_modalities: List of all modality names in order
        Other args: Generation parameters
        
    Returns:
        dict: Generated images for target modalities + condition images
    """
    # Handle random seed
    current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
    
    # Total number of modalities
    num_modalities = len(all_modalities)
    
    with torch.inference_mode():
        # Start with noise for all modalities
        single_noise = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=current_seed)
        x = single_noise.repeat(num_modalities, 1, 1, 1)  # [num_modalities, c, h, w]
        
        # Replace condition modality positions with encoded condition images
        condition_idx = 0
        for i, modality in enumerate(all_modalities):
            if modality in condition_modalities:
                # Encode condition image
                img = condition_images[condition_idx]
                img_resized = img.resize((width, height))
                img_array = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
                
                # Encode to latent space
                with torch.no_grad():
                    latent = vae.encode(img_tensor.to(torch.float32))  # [1, c, h, w] in latent space
                    
                # Replace the noise with condition latent
                x[i] = latent.squeeze(0).clone()  # Clone to avoid reference issues
                
                # Clean up immediately
                del latent, img_tensor
                condition_idx += 1
        
        # Get timesteps
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        
        torch.manual_seed(current_seed)
        
        # Prepare conditioning - replicate prompt for all modalities
        prompts = [prompt] * num_modalities
        neg_prompts = [neg_prompt if neg_prompt else ""] * num_modalities
        
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompts)
        neg_inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=neg_prompts)
        
        # Modified denoising: Use standard denoise but with conditional timesteps
        # We'll manually handle the condition by setting certain modalities to have timestep=0
        
        # Create a custom denoise loop that respects conditions
        x_denoised = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        
        for i, timestep in enumerate(timesteps):
            # Create timestep tensor - 0 for conditions, current timestep for targets
            t_batch = torch.full((num_modalities,), timestep, device=device, dtype=torch.float32)
            for j, modality in enumerate(all_modalities):
                if modality in condition_modalities:  # Set timestep to 0 for condition modalities
                    t_batch[j] = 0.0
            
            # Forward pass
            with torch.no_grad():
                guidance_vec = torch.full((x_denoised.shape[0],), 1, device=device, dtype=x_denoised.dtype)
                
                model_pred = model(
                    img=x_denoised.to(torch.bfloat16),
                    img_ids=inp_cond['img_ids'].to(torch.bfloat16),
                    txt=inp_cond['txt'].to(torch.bfloat16),
                    txt_ids=inp_cond['txt_ids'].to(torch.bfloat16),
                    y=inp_cond['vec'].to(torch.bfloat16),
                    timesteps=t_batch.to(torch.bfloat16),
                    guidance=guidance_vec.to(torch.bfloat16),
                )
                
                # Only apply update to target modalities
                next_timestep = timesteps[i + 1] if i + 1 < len(timesteps) else 0.0
                dt = timestep - next_timestep
                
                for j, modality in enumerate(all_modalities):
                    if modality not in condition_modalities:
                        # Apply Euler step only to target modalities
                        x_denoised[j] = x_denoised[j] - dt * model_pred[j]
                
                # Clean up intermediate tensors
                del model_pred, guidance_vec, t_batch
        
        # Decode final results
        x_final = unpack(x_denoised.float(), height, width)
        x_decoded = vae.decode(x_final)
        x_decoded = x_decoded.clamp(-1, 1)
        
        # Convert to PIL images
        output_imgs = []
        for i in range(num_modalities):
            x_modal = rearrange(x_decoded[i], "c h w -> h w c").cpu()
            output_img = Image.fromarray((127.5 * (x_modal + 1.0)).byte().numpy())
            output_imgs.append(output_img)
            del x_modal
        
        # Clean up all tensors
        del x_decoded, x_denoised, x_final, x, single_noise
        del inp_cond, neg_inp_cond
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create results dict
    results = {}
    for i, modality in enumerate(all_modalities):
        results[modality] = output_imgs[i]
    
    results['all'] = output_imgs
    return results


def run_conditional_inference(config):
    """Run conditional inference using dataset samples - NO METRICS VERSION"""
    # Setup model
    clip, t5, ae, model = setup_lumix_model(config)
    device = config.get('device', 'cuda')
    
    # Create dataset (same as training)
    dataset_config = config.get('dataset_config', {})
    dataset = HypersimDataset(**dataset_config)
    
    # Get evaluation parameters
    condition_modalities = config.get('condition_modalities', ['color'])
    target_modalities = config.get('target_modalities', ['diffuse_reflectance', 'diffuse_illumination', 'depth', 'normal'])
    all_modalities = config.get('all_modalities', ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal'])
    
    # Generation parameters
    width = config.get('width', 512)
    height = config.get('height', 512)
    guidance = config.get('guidance', 4.0)
    num_steps = config.get('num_steps', 50)
    seed = config.get('seed', 42)
    true_gs = config.get('true_gs', 3.5)
    neg_prompt = config.get('neg_prompt', '')
    num_images_per_prompt = config.get('num_images_per_prompt', 1)
    
    # Evaluation parameters
    num_samples = config.get('num_samples', 50)  # Number of samples to evaluate
    save_path = config.get('save_path', 'conditional_inference_results')
    
    print("=" * 60)
    print(f"Conditional Inference Configuration (NO METRICS):")
    print(f"  - Dataset: {dataset_config.get('data_root', 'N/A')}")
    print(f"  - Dataset size: {len(dataset)} samples")
    print(f"  - Samples to process: {num_samples}")
    print(f"  - Condition modalities: {condition_modalities}")
    print(f"  - Target modalities: {target_modalities}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Guidance: {guidance}, True GS: {true_gs}")
    print(f"  - Steps: {num_steps}, Seed: {seed} {'(random)' if seed == -1 else ''}")
    print(f"  - Images per prompt: {num_images_per_prompt}")
    if neg_prompt:
        print(f"  - Negative prompt: {neg_prompt}")
    print("=" * 60)
    
    # Create save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'condition'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'combined'), exist_ok=True)
    
    # Random sample indices - fix progress display issue
    actual_num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), actual_num_samples)
    
    print(f"\nProcessing {actual_num_samples} samples...")
    
    successful_samples = 0
    for i, sample_idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{actual_num_samples} (dataset index: {sample_idx})")
        
        try:
            # Load sample from dataset
            sample = dataset[sample_idx]
            images_dict = sample['images']  # Dict with modality -> PIL Image or tensor
            caption = sample['caption']
            sample_id = sample['sample_id']
            
            # Extract condition images
            condition_images = []
            for modality in condition_modalities:
                if modality in images_dict:
                    img_data = images_dict[modality]
                    
                    # Convert tensor to PIL Image if needed
                    if isinstance(img_data, torch.Tensor):
                        # Convert from tensor to PIL image
                        img_tensor_cpu = img_data.cpu()
                        if img_tensor_cpu.dim() == 3:  # [C, H, W]
                            img_array = ((img_tensor_cpu.permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).byte().numpy()
                        else:  # [H, W, C] or other format
                            img_array = ((img_tensor_cpu + 1.0) * 127.5).clamp(0, 255).byte().numpy()
                        img_pil = Image.fromarray(img_array)
                        condition_images.append(img_pil)
                    elif isinstance(img_data, Image.Image):
                        condition_images.append(img_data)
                    else:
                        # Assume it's a numpy array
                        if img_data.max() <= 1.0:
                            img_array = (img_data * 255).astype(np.uint8)
                        else:
                            img_array = img_data.astype(np.uint8)
                        img_pil = Image.fromarray(img_array)
                        condition_images.append(img_pil)
                else:
                    print(f"Warning: Condition modality {modality} not found in sample {sample_id}")
                    continue
            
            if len(condition_images) != len(condition_modalities):
                print(f"Skipping sample {sample_id}: Missing condition modalities")
                continue
            
            # Generate multiple images per sample if num_images_per_prompt > 1
            for img_idx in range(num_images_per_prompt):
                # Determine output suffix for multiple generations
                if num_images_per_prompt > 1:
                    output_suffix = f"_{img_idx}"
                    print(f"  Generating variant {img_idx+1}/{num_images_per_prompt}...")
                else:
                    output_suffix = ""
                
                # Use different seed for each generation if seed is random
                current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
                
                # Generate conditional images
                print(f"  Generating with prompt: {caption[:60]}... (seed={current_seed})")
                generated_results = generate_conditional_multimodal_images(
                    model, ae, t5, clip, device, caption,
                    condition_images, condition_modalities, target_modalities, all_modalities,
                    width, height, num_steps, guidance, current_seed, true_gs, neg_prompt
                )
                
                # Save generated images (target modalities only)
                for modality in target_modalities:
                    if modality in generated_results:
                        pred_img = generated_results[modality]
                        pred_img.save(os.path.join(save_path, 'generated', f'{sample_id}{output_suffix}_{modality}.png'))
                        print(f"    ✓ Generated {modality}")
                
                # Save condition images for reference (only once for the first generation)
                if img_idx == 0:
                    for j, modality in enumerate(condition_modalities):
                        condition_images[j].save(os.path.join(save_path, 'condition', f'{sample_id}_{modality}.png'))
                
                # Create combined visualization (condition + generated)
                create_combined_image_no_gt(
                    generated_results, condition_images, condition_modalities, target_modalities,
                    os.path.join(save_path, 'combined', f'{sample_id}{output_suffix}_combined.png')
                )
                
                # Clean up after each generation
                del generated_results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"  ✓ Sample {sample_id} completed successfully ({num_images_per_prompt} variant(s) generated)")
            successful_samples += 1
        
        except Exception as e:
            import traceback
            print(f"Error processing sample {sample_idx}: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            continue
    
    print(f"\n✓ Successfully processed {successful_samples}/{actual_num_samples} samples")
    
    # Save generation summary
    summary = {
        'config': config,
        'generation_summary': {
            'total_samples_requested': num_samples,
            'actual_samples_processed': successful_samples,
            'dataset_size': len(dataset),
            'condition_modalities': condition_modalities,
            'target_modalities': target_modalities
        },
        'note': 'No metrics computed - generation only'
    }
    
    summary_file = os.path.join(save_path, 'generation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generation completed! Results saved to: {save_path}")
    print(f"✓ Summary saved to: {summary_file}")
    
    return summary


def create_combined_image_no_gt(generated_results, condition_images, condition_modalities, 
                               target_modalities, output_path):
    """Create combined visualization showing conditions and generated results (no GT needed)"""
    
    # Order: conditions, then generated targets
    combined_images = []
    labels = []
    
    # Add condition images
    for i, modality in enumerate(condition_modalities):
        if i < len(condition_images):
            combined_images.append(condition_images[i])
            labels.append(f"{modality} (condition)")
    
    # Add generated target images
    for modality in target_modalities:
        if modality in generated_results:
            combined_images.append(generated_results[modality])
            labels.append(f"{modality} (generated)")
    
    if not combined_images:
        print("No images to combine")
        return
    
    # Create combined image
    img_width, img_height = combined_images[0].size
    combined_width = img_width * len(combined_images)
    combined_height = img_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    for i, img in enumerate(combined_images):
        combined_img.paste(img, (i * img_width, 0))
    
    combined_img.save(output_path)
    print(f"    Combined image saved: {output_path}")
    print(f"    Layout: {' | '.join(labels)}")


def create_combined_image_with_gt(generated_results, gt_images, condition_modalities, 
                                target_modalities, all_modalities, output_path):
    """Create combined visualization showing conditions, generated, and ground truth"""
    
    # Order: conditions, generated targets, gt targets
    combined_images = []
    labels = []
    
    # Add condition images
    for modality in condition_modalities:
        if modality in generated_results:
            combined_images.append(generated_results[modality])
            labels.append(f"{modality} (cond)")
    
    # Add generated target images
    for modality in target_modalities:
        if modality in generated_results:
            combined_images.append(generated_results[modality])
            labels.append(f"{modality} (gen)")
    
    # Add ground truth target images
    for modality in target_modalities:
        if modality in gt_images:
            combined_images.append(gt_images[modality])
            labels.append(f"{modality} (gt)")
    
    if not combined_images:
        print("No images to combine")
        return
    
    # Create combined image
    img_width, img_height = combined_images[0].size
    combined_width = img_width * len(combined_images)
    combined_height = img_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    for i, img in enumerate(combined_images):
        combined_img.paste(img, (i * img_width, 0))
    
    combined_img.save(output_path)
    print(f"Combined image saved: {output_path} ({', '.join(labels)})")


def run_inference(config):
    """Run LumiX inference based on configuration"""
    # Setup LumiX model
    clip, t5, ae, model = setup_lumix_model(config)
    
    # Get device
    device = config.get('device', 'cuda')
    num_modalities = config.get('num_modalities', 5)

    # Create save directories
    save_path = config.get('save_path', 'lumix_conditional_inference_results')
    os.makedirs(save_path, exist_ok=True)
    
    save_separate = config.get('save_separate', True)
    save_combined = config.get('save_combined', True)
    
    # Modalities from config or default
    modalities = config.get('modalities', ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal'])
    
    if save_separate:
        for modality in modalities:
            modality_dir = os.path.join(save_path, modality)
            os.makedirs(modality_dir, exist_ok=True)
    
    if save_combined:
        combined_dir = os.path.join(save_path, 'combined')
        os.makedirs(combined_dir, exist_ok=True)

    # Get generation parameters
    width = config.get('width', 512)
    height = config.get('height', 512) 
    guidance = config.get('guidance', 4.0)
    num_steps = config.get('num_steps', 50)
    seed = config.get('seed', 42)
    true_gs = config.get('true_gs', 3.5)
    neg_prompt = config.get('neg_prompt', '')  # Support negative prompts
    num_images_per_prompt = config.get('num_images_per_prompt', 1)

    # Process prompts
    prompts_json = config.get('prompts_json')
    single_prompt = config.get('prompt')
    inference_prompts = config.get('inference_prompts', [])
    
    print("=" * 60)
    print(f"LumiX Inference Configuration:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Guidance: {guidance}, True GS: {true_gs}")
    print(f"  - Steps: {num_steps}, Seed: {seed} {'(random)' if seed == -1 else ''}")
    print(f"  - Modalities: {', '.join(modalities)}")
    print(f"  - Images per prompt: {num_images_per_prompt}")
    if neg_prompt:
        print(f"  - Negative prompt: {neg_prompt}")
    print("=" * 60)
    
    if prompts_json and os.path.exists(prompts_json):
        # Load prompts from JSON file
        print(f"Loading prompts from JSON file: {prompts_json}")
        with open(prompts_json, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        print(f"Found {len(prompts_data)} prompts")
        
        for i, item in enumerate(prompts_data):
            image_name = item['image']
            prompt_text = item['text']
            
            print(f"\n[{i+1}/{len(prompts_data)}] Processing: {image_name}")
            print(f"  Prompt: {prompt_text[:80]}...")
            
            # Generate images for this prompt
            for j in range(num_images_per_prompt):
                current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
                print(f"  Generating set {j+1}/{num_images_per_prompt} (seed={current_seed})...")
                
                # Use direct generation (same as training validation)
                results = generate_multimodal_images(
                    model=model,
                    vae=ae,
                    t5=t5,
                    clip=clip,
                    device=device,
                    prompt=prompt_text,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    guidance=guidance,
                    seed=current_seed,
                    true_gs=true_gs,
                    num_modalities=num_modalities,
                    neg_prompt=neg_prompt
                )
                
                # Save results
                processed_name = process_image_name(image_name)
                if num_images_per_prompt > 1:
                    base_name = f"{processed_name}_{j}"
                else:
                    base_name = processed_name
                
                # Save separate modalities to their respective folders
                if save_separate:
                    modalities_dict = results.get('modalities', {}) if hasattr(results, 'get') else results
                    for modality in modalities:
                        img = modalities_dict.get(modality) or (results.get(modality) if hasattr(results, 'get') else None)
                        if img is not None:
                            # Save to modality-specific folder
                            modality_dir = os.path.join(save_path, modality)
                            os.makedirs(modality_dir, exist_ok=True)
                            output_filename = f"{base_name}_{modality}.png"
                            output_path = os.path.join(modality_dir, output_filename)
                            img.save(output_path)
                            print(f"    ✓ {modality} saved")
                
                # Save combined image
                if save_combined:
                    combined_dir = os.path.join(save_path, 'combined')
                    os.makedirs(combined_dir, exist_ok=True)
                    combined_filename = f"{base_name}_combined.png"
                    combined_path = os.path.join(combined_dir, combined_filename)
                    create_combined_image(results, combined_path, modality_order=modalities)
                
                # Optionally increment seed for variety
                # seed += 1
                
    elif single_prompt:
        # Single prompt from config
        print(f"Processing single prompt: {single_prompt}")
        
        for i in range(num_images_per_prompt):
            current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
            print(f"\nGenerating set {i+1}/{num_images_per_prompt} (seed={current_seed})...")
            
            # Use direct generation (same as training validation)
            results = generate_multimodal_images(
                model=model,
                vae=ae,
                t5=t5,
                clip=clip,
                device=device,
                prompt=single_prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance=guidance,
                seed=current_seed,
                true_gs=true_gs,
                num_modalities=num_modalities,
                neg_prompt=neg_prompt
            )
            
            # Save results
            safe_prompt = re.sub(r'[<>:"/\\|?*]', '_', single_prompt.replace(" ", "_"))
            prompt_path = safe_prompt[:100]
            
            full_prompt_dir = os.path.join(save_path, prompt_path)
            os.makedirs(full_prompt_dir, exist_ok=True)
            
            # Save images to modality-specific folders
            if save_separate:
                modalities_dict = results.get('modalities', {}) if hasattr(results, 'get') else results
                for modality in modalities:
                    img = modalities_dict.get(modality) or (results.get(modality) if hasattr(results, 'get') else None)
                    if img is not None:
                        # Save to modality-specific folder
                        modality_dir = os.path.join(save_path, modality)
                        os.makedirs(modality_dir, exist_ok=True)
                        output_filename = f"result_{i}_{modality}.png"
                        output_path = os.path.join(modality_dir, output_filename)
                        img.save(output_path)
                        print(f"  ✓ {modality} saved")
            else:
                # Save to prompt-specific folder
                modalities_dict = results.get('modalities', {}) if hasattr(results, 'get') else results
                for modality in modalities:
                    img = modalities_dict.get(modality) or (results.get(modality) if hasattr(results, 'get') else None)
                    if img is not None:
                        output_filename = f"result_{i}_{modality}.png"
                        output_path = os.path.join(full_prompt_dir, output_filename)
                        img.save(output_path)
                        print(f"  ✓ {modality} saved")
            
            # Save combined image
            if save_combined:
                combined_dir = os.path.join(save_path, 'combined')
                os.makedirs(combined_dir, exist_ok=True)
                combined_filename = f"result_{i}_combined.png"
                combined_path = os.path.join(combined_dir, combined_filename)
                create_combined_image(results, combined_path, modality_order=modalities)
            
            # seed += 1
            
    elif inference_prompts:
        # Multiple prompts from config
        print(f"Processing {len(inference_prompts)} prompts from config")
        
        for i, prompt_text in enumerate(inference_prompts):
            print(f"\n[{i+1}/{len(inference_prompts)}] Prompt: {prompt_text[:60]}...")
            
            for j in range(num_images_per_prompt):
                current_seed = seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
                print(f"  Generating set {j+1}/{num_images_per_prompt} (seed={current_seed})...")
                
                # Use direct generation (same as training validation)
                results = generate_multimodal_images(
                    model=model,
                    vae=ae,
                    t5=t5,
                    clip=clip,
                    device=device,
                    prompt=prompt_text,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    guidance=guidance,
                    seed=current_seed,
                    true_gs=true_gs,
                    num_modalities=num_modalities,
                    neg_prompt=neg_prompt
                )
                
                # Save results
                safe_prompt = re.sub(r'[<>:"/\\|?*]', '_', prompt_text.replace(" ", "_"))
                prompt_path = safe_prompt[:50]  # Shorter for multiple prompts
                
                full_prompt_dir = os.path.join(save_path, f"prompt_{i+1}_{prompt_path}")
                os.makedirs(full_prompt_dir, exist_ok=True)
                
                # Save images to modality-specific folders
                if save_separate:
                    modalities_dict = results.get('modalities', {}) if hasattr(results, 'get') else results
                    for modality in modalities:
                        img = modalities_dict.get(modality) or (results.get(modality) if hasattr(results, 'get') else None)
                        if img is not None:
                            # Save to modality-specific folder
                            modality_dir = os.path.join(save_path, modality)
                            os.makedirs(modality_dir, exist_ok=True)
                            output_filename = f"prompt_{i+1}_{j}_{modality}.png"
                            output_path = os.path.join(modality_dir, output_filename)
                            img.save(output_path)
                            print(f"    ✓ {modality} saved")
                else:
                    # Save to prompt-specific folder
                    modalities_dict = results.get('modalities', {}) if hasattr(results, 'get') else results
                    for modality in modalities:
                        img = modalities_dict.get(modality) or (results.get(modality) if hasattr(results, 'get') else None)
                        if img is not None:
                            output_filename = f"result_{j}_{modality}.png"
                            output_path = os.path.join(full_prompt_dir, output_filename)
                            img.save(output_path)
                            print(f"    ✓ {modality} saved")
                
                # Save combined image
                if save_combined:
                    combined_dir = os.path.join(save_path, 'combined')
                    os.makedirs(combined_dir, exist_ok=True)
                    combined_filename = f"prompt_{i+1}_{j}_combined.png"
                    combined_path = os.path.join(combined_dir, combined_filename)
                    create_combined_image(results, combined_path, modality_order=modalities)
                
                # seed += 1
    else:
        print("Error: No prompts specified in config. Please set 'prompt', 'prompts_json', or 'inference_prompts'")
        return

    print("\n" + "=" * 60)
    print("✓ All LumiX multi-modal images generated successfully!")
    print(f"✓ Results saved to: {save_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LumiX Multi-modal Conditional Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", type=str, nargs='*', help="Override config values (e.g., --override device=cpu seed=42)")
    parser.add_argument("--mode", type=str, choices=['conditional', 'standard'], default='conditional', 
                       help="Inference mode: conditional (using dataset) or standard (prompt-based)")
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        return
    
    config = load_config(args.config)
    print(f"✓ Loaded config from: {args.config}")
    
    # Apply overrides
    if args.override:
        print("\nApplying config overrides:")
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                except:
                    pass  # Keep as string
                config[key] = value
                print(f"  - {key} = {value}")
    
    # Run inference based on mode
    if args.mode == 'conditional':
        print("\n🎯 Running CONDITIONAL inference mode")
        run_conditional_inference(config)
    else:
        print("\n🎨 Running STANDARD inference mode")
        run_inference(config)


if __name__ == "__main__":
    main()
