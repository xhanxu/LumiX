# Base Models

Optional local cache for base model weights.

By default, the inference scripts download or load the required FLUX/text-encoder components through Hugging Face utilities. If you already have local copies, you can place them here and point environment variables such as:

```bash
export FLUX_DEV="models/flux/flux1-dev.safetensors"
export AE="models/flux/ae.safetensors"
```

The base model files are not included in this inference release.
