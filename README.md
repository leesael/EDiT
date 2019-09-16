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

You should first download the dataset from [this website](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/) and place the directory as `/data`.
Then, move to `/src` and run `main.py` to run EDiT.
Currently it trains a vanilla SDT over the `abalone` dataset, but you can change easily the hyperparameters in `main.py` including the dataset, sparsification technique, and training procedure.
```
python main.py
```

You need, however, pretrained random forest (RF) models to use the knowledge distillation.

## References

```
@inproceedings{YooS19,
  author    = {Jaemin Yoo and Lee Sael},
  title     = {EDiT: Interpreting Ensemble Models via Compact Soft Decision Trees},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2019}
}
```
