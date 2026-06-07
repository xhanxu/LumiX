# Data

Optional input data for conditional inference.

Expected example layout:

```text
data/
  suppl_test/
    split1/
      color/
        example_color.png
      captions_blip2/
        example_color.json
```

Update `dataset_config.data_root` and `dataset_config.captions_dir` in:

```text
eval/configs/lumix_conditional_inference.yaml
```
