from pandas import DataFrame
import pickle
import logging
from typing import List
from sklearn.model_selection import GridSearchCV


log = logging.Logger('logger')
log.setLevel('INFO')


def init_model() -> GridSearchCV:
    """ML model initialization with .pkl file
    """
    log.info('Initializing the model from the pickle file ...')
    with open('source/model_gscv_ridge.pkl', 'rb') as file:
        model = pickle.load(file)
    log.info('Model is imported successfully!')

    return model


def scorer(df: DataFrame) -> List:
    """Make predictions on the input data
    """
    model = init_model()
    log.info('Scoring input objects ...')
    predictions_arr = model.predict(df)
    log.info('Predictions vector is ready!')

    return predictions_arr
