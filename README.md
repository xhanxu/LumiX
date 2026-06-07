# LumiX ✨

Official code for **LumiX: Structured and Coherent Text-to-Intrinsic Generation**.

🌐 Project page: [https://xhanxu.github.io/lumix.github.io/](https://xhanxu.github.io/lumix.github.io/)  
📄 Paper: [arXiv:2512.02781](https://arxiv.org/abs/2512.02781)  
🏆 Accepted to **CVPR 2026**

LumiX predicts aligned visual modalities such as:

- diffuse reflectance
- diffuse illumination
- color
- depth
- normal

LumiX is built from two pieces:

- **QBA, Query Broadcast Attention** 🛰️  
  The color query is broadcast across modalities so generated outputs stay spatially aligned.

- **Tensor LoRA** 🧩  
  A compact cross-modality LoRA adapter that lets each output modality aggregate information from all modalities through a tensorized low-rank structure.

Model checkpoints are not included in this repository.

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

LumiX checkpoints will be released on Hugging Face:

👉 [https://huggingface.co/hanx/lumix](https://huggingface.co/hanx/lumix)

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

The LumiX training code will be released in a future update, including scripts and configs for training the QBA + Tensor LoRA adapters.

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
