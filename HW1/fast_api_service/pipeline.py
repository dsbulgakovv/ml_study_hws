import pandas as pd
from pandas import DataFrame
from datetime import datetime
import numpy as np
import re
import logging

log = logging.Logger('logger')
log.setLevel('INFO')

NUM_FEATURES = [
    'km_driven', 'mileage', 'engine', 'max_power', 'torque_nm',
    'torque_rpm', 'car_age', 'age_sq', 'combo_power', 'torque_rpm_log'
    ]
CAT_FEATURES = [
    'fuel', 'seller_type', 'transmission', 'owner', 'brand_class', 'seats'
       ]


def torque(x: str) -> float:
    """Extract value of torque and convert it to Nm
    """
    val = str(x).lower()
    if 'kgm' in val:
        val_spl = val.replace(',', '.')
        reg = re.findall(r'(\d+[.,]*\d*)', val_spl)[0]
        res = float(reg) * 9.80665
    elif ('nm' in val) or (val != 'nan'):
        val_spl = val.replace(',', '.')
        reg = re.findall(r'(\d+[.,]*\d*)', val_spl)[0]
        res = float(reg)
    else:
        res = np.nan

    return res


def torque_rpm(x: str) -> float:
    """ Extract value of rounds per minute """
    val = str(x).lower()
    if val != 'nan':
        reg = re.findall(r'(\d+)', val)[-1]
        res = float(reg)
    else:
        res = np.nan

    return res


def fill_empty(df: DataFrame) -> DataFrame:
    imp = ...
    df[NUM_FEATURES] = imp.transform(df[NUM_FEATURES])

    return df


def change_cat_features(df: DataFrame) -> DataFrame:
    """Change input dataframe columns: mileage, engine, max_power
    and make 2 new columns: torque_nm and torque_rpm from torque.
    """
    df['mileage'] = (
        df['mileage']
        .astype(str)
        .str.extract(r'(\d+[.,]*\d*)')
        .astype(float)
    )
    df['engine'] = (
        df['engine']
        .astype(str)
        .str.extract(r'(\d+[.,]*\d*)')
        .astype(float)
    )
    df['max_power'] = (
        df['max_power']
        .astype(str)
        .str.extract(r'(\d+[.,]*\d*)')
        .astype(float)
    )
    df['torque_nm'] = df['torque'].apply(torque).astype(int)
    df['torque_rpm'] = df['torque'].apply(torque_rpm).astype(int)
    df.drop('torque', axis=1, inplace=True)

    return df


def add_brand(df: DataFrame) -> DataFrame:
    """Add brand class feature to dataframe
    """
    brands_df = pd.read_csv('source/brands_classes.csv')
    out_df = (
        df.join(brands_df.set_index('brand'), ['brand'], 'left')
    )
    out_df.drop('name', axis=1, inplace=True)

    return out_df


def add_car_age(df: DataFrame) -> DataFrame:
    """Add age feature
    """
    cur_year = datetime.today().year
    df['car_age'] = df['year'].apply(lambda x: int(cur_year - int(x)))
    df.drop('year', axis=1, inplace=True)

    return df


def add_custom_feature(df: DataFrame) -> DataFrame:
    """Add custom feature like in formula: CC / hsp * log2(rpm)
    strange feature :)
    """
    df['combo_power'] = (
            df['engine'] / df['max_power'] *
            df['torque_rpm'].apply(lambda x: np.log2(x))
    ).apply(lambda x: round(x, 8))

    return df


def add_polynom(df: DataFrame) -> DataFrame:
    rows_ini = df.shape[0]
    df = df[np.isfinite(df[NUM_FEATURES]).all(1)]
    df.reset_index(drop=True, inplace=True)
    rows_aft = df.shape[0]
    if rows_aft < rows_ini:
        log.info(f'{rows_ini - rows_aft} rows are excluded because of infinite feature value.')
    poly = ...
    polynom_df = pd.DataFrame(
        poly.transform(df[NUM_FEATURES]),
        columns=poly.get_feature_names_out()
    )
    out_df = df.drop(NUM_FEATURES, axis=1).join(polynom_df)

    return out_df


def add_ohe(df: DataFrame) -> DataFrame:
    ohe = ...
    ohe_df = pd.DataFrame.sparse.from_spmatrix(
        ohe.transform(df[CAT_FEATURES]),
        columns=ohe.get_feature_names_out()
    )
    out_df = df.drop(CAT_FEATURES, axis=1).join(ohe_df)

    return out_df


def add_scaler(df: DataFrame) -> DataFrame:
    scaler = ...
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

    return df
