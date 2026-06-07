# Checkpoints

Place LumiX LoRA checkpoints here.

Model checkpoints will be released on Hugging Face:

https://huggingface.co/hanx/lumix0

Example download command:

```bash
huggingface-cli download hanx/lumix0 lora_color_enhanced.safetensors \
  --local-dir checkpoints/lumix
```

Recommended layout:

```text
checkpoints/
  lumix/
    lora_color_enhanced.safetensors
```

Then update the YAML config:

```yaml
lora_local_path: "checkpoints/lumix/lora_color_enhanced.safetensors"
```

Large checkpoint files are hosted on Hugging Face and are intentionally not committed to this repository.
