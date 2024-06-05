# Masked Graph Augmentation for Graph Contrastive Learning

This repository studies the effectiveness of graph augmentation for graph representation learning.
We have a load of observations.
This paper is *Under Review*.

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
