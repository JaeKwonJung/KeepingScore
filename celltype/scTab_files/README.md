# scTab resources for Keeping SCORE

This directory collects the external [scTab](https://github.com/theislab/scTab)
resources used by the **Keeping SCORE** cell-type workflows: the Merlin dataset,
pretrained scTab checkpoints, the upstream scTab source, and a local adapter for
training classifiers on precomputed embeddings.

> [!IMPORTANT]
> The compressed downloads are approximately **164 GB** for the Merlin dataset
> and **8.1 GB** for the checkpoints. Allow at least **220 GB** for the extracted
> resources—and substantially more if retaining both archives during extraction.

## Contents

| Path | Purpose | Included in this repository? |
| --- | --- | :---: |
| [`emb_cellnet/`](./emb_cellnet) | Embedding-aware versions of the scTab estimator and classifier models | Yes |
| `merlin_cxg_2023_05_15_sf-log1p/` | Merlin dataloader files and cell-type metadata | No — download required |
| `scTab-checkpoints/` | Pretrained scTab model checkpoints | No — download required |
| `scTab-devel/` | Upstream scTab source tree | No — clone required |

## Quick setup

The Merlin dataset and pretrained checkpoints **must be downloaded manually**
from the links below (for example, with a local web browser) and then uploaded
to `KeepingScore/celltype/scTab_files/`. Do not expect the `curl` commands to
work from the compute environment.

- [Download the Merlin dataset (~164 GB)](https://pklab.med.harvard.edu/felix/data/merlin_cxg_2023_05_15_sf-log1p.tar.gz)
- [Download the pretrained scTab checkpoints (~8.1 GB)](https://pklab.med.harvard.edu/felix/data/scTab-checkpoints.tar.gz)

After uploading both archives, their paths should be:

```text
KeepingScore/celltype/scTab_files/merlin_cxg_2023_05_15_sf-log1p.tar.gz
KeepingScore/celltype/scTab_files/scTab-checkpoints.tar.gz
```

Then, from the `KeepingScore` repository root, extract them and clone scTab:

```bash
cd celltype/scTab_files

# 1. Extract the manually downloaded and uploaded Merlin dataset.
tar -xzf merlin_cxg_2023_05_15_sf-log1p.tar.gz

# 2. Extract the manually downloaded and uploaded pretrained checkpoints.
tar -xzf scTab-checkpoints.tar.gz

# 3. Clone the upstream scTab source into the name expected by this project.
git clone --branch devel --single-branch https://github.com/theislab/scTab.git scTab-devel
```

## Expected layout

After setup, the relevant files should be arranged as follows:

```text
scTab_files/
├── emb_cellnet/
│   ├── estimators.py
│   └── models.py
├── merlin_cxg_2023_05_15_sf-log1p/
│   ├── categorical_lookup/
│   │   └── cell_type.parquet
│   └── ...
├── scTab-checkpoints/
│   └── scTab/
│       └── run4/
│           └── val_f1_macro_epoch=45_val_f1_macro=0.847.ckpt
└── scTab-devel/
    ├── cellnet/
    └── notebooks/
```

Do not rename these directories without also updating the relative paths in the
cell-type scripts and notebooks.

## Verify the installation

From `scTab_files/`, this small check reports whether the key resources expected
by the current workflows are present:

```bash
for path in \
  merlin_cxg_2023_05_15_sf-log1p/categorical_lookup/cell_type.parquet \
  merlin_cxg_2023_05_15_sf-log1p/cell_type_hierarchy/child_matrix.npy \
  merlin_cxg_2023_05_15_sf-log1p/class_weights.npy \
  scTab-checkpoints/scTab/run4/val_f1_macro_epoch=45_val_f1_macro=0.847.ckpt \
  scTab-devel/notebooks/data_augmentation/shortend_cell_types.yaml
do
  if [[ -e "$path" ]]; then
    printf '✓ %s\n' "$path"
  else
    printf '✗ missing: %s\n' "$path"
  fi
done
```

## About `emb_cellnet`

`emb_cellnet` is derived from scTab's `cellnet` implementation and adapted for
classifiers that consume extracted embeddings rather than raw single-cell gene
expression matrices. In particular, its estimator supports `embedding=True` and
uses a caller-provided embedding data module while retaining scTab's label
metadata, class weights, hierarchy, and model components.

The adapter still imports modules from the sibling `scTab-devel/cellnet` source
tree, so the upstream clone is required even when working only with embeddings.

## Upstream resources

- [scTab source and documentation](https://github.com/theislab/scTab)
- [Merlin dataset archive](https://pklab.med.harvard.edu/felix/data/merlin_cxg_2023_05_15_sf-log1p.tar.gz)
- [Pretrained scTab checkpoints](https://pklab.med.harvard.edu/felix/data/scTab-checkpoints.tar.gz)
- [Keeping SCORE cell-type workflow](../README.md)

External data, checkpoints, and source code remain subject to their respective
upstream terms and licenses.
