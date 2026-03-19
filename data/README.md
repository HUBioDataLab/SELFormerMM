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
| `models/` | pretrained SELFormerMM model + DMGI checkpoint for KG embedding. |
| `finetuned_models/` | fine-tuned molecular property prediction task checkpoints |


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
├── models/
│   ├── DMGI/
│   │   └── dmgi_model.pt
│   └── SELFormerMM/       → config, tokenizer, pytorch_model.bin (weights)
└── finetuned_models/      → per-task fine-tuned checkpoints (bbbp, esol, …)
```


