import pandas as pd
import logging
from pandas import DataFrame
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from io import BytesIO

from feature_pipeline import full_pipeline
from model import scorer


log = logging.Logger('logger')
log.setLevel('INFO')

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get('/')
def root():
    """Show welcome text phrase
    """
    out_dict = dict()
    out_dict['message'] = \
        """
        Hello!
        You can use post queries with .../predict_item or .../predict_items.
        Just run one of the commands above and send JSON string or .csv file.
        """

    return out_dict


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """Make a prediction on one item
    """
    log.info('Turning JSON into one-row-df ...')
    item_dict = item.model_dump()
    item_lists = dict()
    for key in item_dict.keys():
        item_lists[key] = list()
        item_lists[key].append(item_dict[key])
    df = pd.DataFrame(item_lists)
    log.info('Engineering features ...')
    fin_df = full_pipeline(df)
    log.info('Making predictions ...')
    predictions = scorer(fin_df)
    log.info('Success!')

    return predictions[0]


def upload_csv(file: UploadFile) -> DataFrame:
    """Upload csv file and return list of dicts with Item entries
    """
    content = file.file.read()  # считываем байтовое содержимое
    buffer = BytesIO(content)  # создаем буфер типа BytesIO
    df = pd.read_csv(buffer)
    buffer.close()
    file.close()
    # df.iloc[0:2].to_json(orient='table', index=False)
    # obj_lst = json.loads(item)['data']

    return df


@app.post("/predict_items")
def predict_items(file: UploadFile) -> List[float]:
    """Make predictions on many items
    """
    df = upload_csv(file)
    log.info('Engineering features ...')
    fin_df = full_pipeline(df)
    log.info('Making predictions ...')
    predictions = scorer(fin_df)
    log.info('Success!')

    return predictions
