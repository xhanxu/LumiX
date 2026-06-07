# Checkpoints

Place LumiX LoRA checkpoints here.

Model checkpoints are hosted on Hugging Face:

https://huggingface.co/hanx/LumiX_ckp

Example download command:

```bash
huggingface-cli download hanx/LumiX_ckp lumix.safetensors \
  --local-dir checkpoints/lumix
```

Recommended layout:

```text
checkpoints/
  lumix/
    lumix.safetensors
```

Then update the YAML config:

```yaml
lora_local_path: "checkpoints/lumix/lumix.safetensors"
```

Large checkpoint files are hosted on Hugging Face and are intentionally not committed to this repository.
