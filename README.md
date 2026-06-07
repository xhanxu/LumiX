# LumiX [CVPR 2026] ✨

Official code for **LumiX: Structured and Coherent Text-to-Intrinsic Generation**.

🌐 [Project page](https://xhanxu.github.io/lumix.github.io/) 
📄 [Paper](https://arxiv.org/abs/2512.02781)  

LumiX is a structured diffusion framework for coherent text-to-intrinsic generation.
Given a text prompt, LumiX jointly generates a comprehensive set of intrinsic maps,
including diffuse reflectance, diffuse illumination, normal, depth, and final color.
These outputs form an aligned and physically consistent description of the same
underlying scene, enabling unified generation of multiple intrinsic properties from
text and image-conditioned intrinsic decomposition within a single framework.


## Install 🛠️

Create a fresh environment:

```bash
conda create -n lumix python=3.10 -y
conda activate lumix
```

Install PyTorch for your CUDA version from the official PyTorch instructions, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you want a one-shot install after PyTorch is already available:

```bash
pip install torch transformers huggingface-hub safetensors einops Pillow PyYAML numpy
```

## Checkpoints 📦

LumiX checkpoints will be released on [Hugging Face](https://huggingface.co/hanx/lumix):

Download the LumiX LoRA checkpoint and place it under `checkpoints/lumix/`:

```bash
huggingface-cli download hanx/lumix0 lora_color_enhanced.safetensors \
  --local-dir checkpoints/lumix
```

Then update the config if your filename or path differs:

```yaml
lora_local_path: "checkpoints/lumix/lora_color_enhanced.safetensors"
```

Recommended local layout:

```text
checkpoints/
  lumix/
    lora_color_enhanced.safetensors
```

The base FLUX and text encoder weights are loaded through Hugging Face by `src/flux/util.py`. You can also point environment variables such as `FLUX_DEV` and `AE` to local checkpoints if you already have them cached:

```text
models/
  flux/
    flux1-dev.safetensors
    ae.safetensors
```

## Text-To-Multimodal Inference 🎨

Edit:

```text
eval/configs/lumix_hypersim_inference.yaml
```

Then run:

```bash
python eval/eval_lumix_hypersim_inference.py \
  --config eval/configs/lumix_hypersim_inference.yaml
```

This generates all five modalities from text prompts listed in the config.

## Conditional Inference 🧭

Edit:

```text
eval/configs/conditional_inference_config_no_metrics_lumix.yaml
```

Set:

```yaml
dataset_config:
  data_root: "data/suppl_test/split1"
  captions_dir: "data/suppl_test/split1/captions_blip2"
  condition_modalities: ["color"]

condition_modalities: ["color"]
target_modalities: ["diffuse_reflectance", "diffuse_illumination", "depth", "normal"]
```

Then run:

```bash
bash eval/run_hypersim_inference.sh
```

The default shell script runs conditional inference from color to the remaining modalities. To run text-to-multimodal inference instead, uncomment the first Python command in `eval/run_hypersim_inference.sh`.

## Repository Layout 🗂️

```text
eval/
  run_hypersim_inference.sh
  eval_lumix_hypersim_inference.py
  eval_lumix_hypersim_inference_condition_no_metrics.py
  configs/
    lumix_hypersim_inference.yaml
    conditional_inference_config_no_metrics_lumix.yaml

src/flux/
  model.py
  sampling.py
  util.py
  modules/
    layers_lumix.py

checkpoints/
  lumix/
    # put LumiX LoRA checkpoints here

models/
  flux/
    # optional local FLUX/AE cache

data/
  # optional conditional inference inputs

training/
  README.md
```

## Training Code 🚀

The LumiX training code will be released in a future update, including scripts and configs.

## Notes 💡

- Checkpoint files are hosted on [Hugging Face](https://huggingface.co/hanx/lumix0). Put them in `checkpoints/lumix/`, or change `lora_local_path`.
- The default configs use `device: "cuda"`. Change this if you want a specific GPU such as `cuda:0`.
- Conditional inference expects images under modality folders such as `color/`, with matching captions in `captions_blip2/`.

## Citation 📚

If you find LumiX useful, please cite:

```bibtex
@inproceedings{han2026lumix,
  title={LumiX: Structured and Coherent Text-to-Intrinsic Generation},
  author={Han, Xu and Zhang, Biao and Tang, Xiangjun and Li, Xianzhi and Wonka, Peter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21942--21952},
  year={2026}
}
```
