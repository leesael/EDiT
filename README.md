# EDiT

This project is a PyTorch implementation of EDiT: Interpreting Ensemble Models via Compact Soft Decision Trees, published as a conference proceeding at [ICDM 2019](http://icdm2019.bigke.org/).
This paper proposes a novel approach that distills the knowledge of an ensemble model to maximize the interpretability of soft decision trees (SDT) with fewer parameters.

## Prerequisites

- Python 3.6+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org/stable/)
- [joblib](https://joblib.readthedocs.io/en/latest/)
- [pandas](https://pandas.pydata.org/)

## Usage

You should first download the datasets from [this website](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/) and place them in `/data`.
You may just run `down.sh` in `data/` in a Linux environment.
Although it contains over a hundred datasets which were used in previous works, we use only 8 of them in our work.
The list of target datasets is described in `datasets.txt`.

Then, move to `src/` and run `python main.py` to actually run EDiT.
Currently it trains a vanilla SDT over the `abalone` dataset, but you can change easily the hyperparameters in `src/main.py` including the dataset, sparsification technique, and training procedure.
For instance, it will use the tree pruning technique if you change `tree_threshold` from `0` to a desired threshold such as `1e-4`.

If you want to enable the knowledge distillation technique, you should run `python rf.py` to train and save random forests (RF) which are not included in this repository.
The trained RF models are saved in `out/rf/models`.
The other results such as intermediate logs of training and trained compact soft decision trees are saved in `out/edit`.

## References

```
@inproceedings{YooS19,
  author    = {Jaemin Yoo and Lee Sael},
  title     = {EDiT: Interpreting Ensemble Models via Compact Soft Decision Trees},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2019}
}
```
