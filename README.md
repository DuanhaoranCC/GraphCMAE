# GraphCMAE

This repository is for the source code of the paper "GraphCMAE: Contrastive Masked AutoEncoder for Self-Supervised Graph Representation Learning".

## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

For graph classification task:
```python
python main.py --dataset COLLAB
```

The `--dataset` argument should be one of [NCI1, NCI109, PROTEIN, DD, MUTAG, IMDB-B, IMDB-M, COLLAB, RDT-B].
