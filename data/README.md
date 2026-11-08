**Data and model download.** Pretraining and fine-tuning datasets (including all raw modalities), single-modality embedding files (`graph` / `text` / `kg`), the pretrained SELFormerMM multimodal checkpoint, and fine-tuned task checkpoints are available from the Hugging Face dataset: [HUBioDataLab/SELFormerMM](https://huggingface.co/datasets/HUBioDataLab/SELFormerMM). After downloading, place the files under this `data/` directory so paths match the [SELFormerMM](https://github.com/HUBioDataLab/SELFormerMM) README and scripts. 


```bash
# Option A: clone the dataset into ./data (adjust if your layout differs)
huggingface-cli download HUBioDataLab/SELFormerMM --repo-type dataset --local-dir data

# Option B: download only a subtree, e.g. pretraining_datasets
huggingface-cli download HUBioDataLab/SELFormerMM pretraining_datasets \
  --repo-type dataset --local-dir data
```

## Contents (summary)

| Area | Role |
|------|------|
| `pretraining_datasets/` | Multimodal pretraining: raw modality CSV + aligned `graph` / `text` / `kg` embedding files + KG `HeteroData`. |
| `finetuning_datasets/` | MoleculeNet benchmark datasets: `{task}.csv` + `{task}_embs.npz` (`graph`, `text`, `kg`). |
| `models/SELFormerMM/` | Pretrained multimodal backbone: config, tokenizer, `pytorch_model.bin`. |
| `models/DMGI/` | DMGI checkpoint used to produce the KG embeddings. |
| `models/finetuned/` | Fine-tuned task checkpoints, three seeds per task, each with the hyperparameter config that produced it. |

### Fine-tuning configs

Each published checkpoint ships a `finetuning_optimized_config.json` next to its weights. It records
the dataset paths, the split settings and the optimization settings that produced that
checkpoint. Pass it to `train_finetuning.py` to retrain the task, or to `evaluate.py` to reproduce the checkpoint's score:

```bash
python3 train_finetuning.py \
  --config=data/models/finetuned/bbbp/seed_42/finetuning_optimized_config.json \
  --save_dir=runs/bbbp

python3 evaluate.py \
  --model_dir=data/models/finetuned/bbbp \
  --config=data/models/finetuned/bbbp/seed_42/finetuning_optimized_config.json \
  --device=cuda
```

The three seeds' configs are identical except for the `seed` field. That field also fixes the split,
so a checkpoint has to be scored with its own config; `evaluate.py` takes the seed from the
`seed_<n>` directory name and rejects a contradicting `--seed`.


## Expected directory tree (`data/`)

```
data/
├── README.md
├── pretraining_datasets/
│   ├── pretraining_dataset_meta.csv
│   ├── graph_embeddings.npy
│   ├── text_embeddings.npy
│   ├── kg_embeddings.npy
│   └── selformermm_kg_heterodata.pt
├── finetuning_datasets/
│   ├── classification/
│   │   ├── bace/          → bace.csv, bace_embs.npz
│   │   ├── bbbp/
│   │   ├── hiv/
│   │   ├── sider/
│   │   └── tox21/
│   └── regression/
│       ├── esol/
│       ├── freesolv/
│       ├── lipo/
│       └── pdbbind_full/
└── models/
    ├── DMGI/
    │   └── dmgi_model.pt
    ├── SELFormerMM/       → config, tokenizer, pytorch_model.bin (weights)
    └── finetuned/
        ├── bace/
        │   ├── seed_42/   → model.pt, finetuning_optimized_config.json, tokenizer
        │   ├── seed_123/
        │   └── seed_3407/
        ├── bbbp/
        ├── esol/
        ├── freesolv/
        ├── hiv/
        ├── lipo/
        ├── pdbbind_full/
        ├── sider/
        └── tox21/
```

Three seeds (42, 123, 3407) are published per task so that the mean ± standard deviation reported in
the paper can be recomputed rather than approximated from a single checkpoint. There is no default
or "recommended" seed; any one of them is a working model.


