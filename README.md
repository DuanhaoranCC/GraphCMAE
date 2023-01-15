# A Simple Self-Supervised Graph Representation 

Self-supervised graph representation learning is a key technique for graph structured data processing, especially for Web-generated graph that do not have qualified labelling information.
## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

For graph classification task:
```python
python main.py --dataset ENZYMES
```

The `--dataset` argument should be one of [ENZYMES, PTC-MR, NCI1, NCI109, PROTEIN, DD, MUTAG, IMDB-B, IMDB-M, COLLAB, RDT-B].
