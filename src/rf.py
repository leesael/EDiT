import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import data


def main():
    dataset = 'abalone'
    data_path = '../data'
    out_path = '../out/rf/models'
    model_path = f'{out_path}/{dataset}.pkl'
    seed = 2019

    np.random.seed(seed)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    data_dict = data.read_data(data_path, dataset, validation=False)
    trn_x = data_dict['trn_x']
    trn_y = data_dict['trn_y']
    model.fit(trn_x, trn_y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == '__main__':
    main()
