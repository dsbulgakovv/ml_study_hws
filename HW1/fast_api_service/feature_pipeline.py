import pandas as pd
from pandas import DataFrame
from datetime import datetime
import numpy as np
import pickle
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
    """Extract value of rounds per minute
    """
    val = str(x).lower()
    if val != 'nan':
        reg = re.findall(r'(\d+)', val)[-1]
        res = float(reg)
    else:
        res = np.nan

    return res


def fill_empty(df: DataFrame, pickl_file_path: str) -> DataFrame:
    with open(pickl_file_path, 'rb') as file:
        imp = pickle.load(file)
    features = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'seats', 'torque_nm', 'torque_rpm']
    df[features] = imp.transform(df[features])

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
    df['brand'] = df['name'].apply(lambda x: str(x).split(' ')[0])
    out_df = (
        df.join(brands_df.set_index('brand'), ['brand'], 'left')
    )
    out_df.drop(['name', 'brand'], axis=1, inplace=True)

    return out_df


def add_car_age(df: DataFrame) -> DataFrame:
    """Add age feature
    """
    cur_year = datetime.today().year
    df['car_age'] = df['year'].apply(lambda x: int(cur_year - int(x)))
    df['age_sq'] = df['car_age'].apply(lambda x: x**2)
    df.drop('year', axis=1, inplace=True)

    return df


def add_custom_feature(df: DataFrame) -> DataFrame:
    """Add custom feature like in formula: CC / hsp * log2(rpm)
    strange feature :)
    """
    df['torque_rpm_log'] = df['torque_rpm'].apply(lambda x: np.log2(x))
    df['combo_power'] = (
            df['engine'] / df['max_power'] * df['torque_rpm_log']
    ).apply(lambda x: round(x, 8))

    return df


def add_polynom(df: DataFrame, pickl_file_path: str) -> DataFrame:
    """Add polynomial features
    """
    rows_ini = df.shape[0]
    df = df[np.isfinite(df[NUM_FEATURES]).all(1)]
    df.reset_index(drop=True, inplace=True)
    rows_aft = df.shape[0]
    if rows_aft < rows_ini:
        log.info(f'{rows_ini - rows_aft} rows are excluded because of infinite feature value.')
    with open(pickl_file_path, 'rb') as file:
        poly = pickle.load(file)
    polynom_df = pd.DataFrame(
        poly.transform(df[NUM_FEATURES]),
        columns=poly.get_feature_names_out()
    )
    out_df = df.drop(NUM_FEATURES, axis=1).join(polynom_df)

    return out_df


def add_ohe(df: DataFrame, pickl_file_path: str) -> DataFrame:
    """Add one hot encoded features
    """
    with open(pickl_file_path, 'rb') as file:
        ohe = pickle.load(file)
    ohe_df = pd.DataFrame.sparse.from_spmatrix(
        ohe.transform(df[CAT_FEATURES]),
        columns=ohe.get_feature_names_out()
    )
    out_df = df.drop(CAT_FEATURES, axis=1).join(ohe_df)

    return out_df


def add_scaler(df: DataFrame, pickl_file_path: str) -> DataFrame:
    """Add scales features
    """
    with open(pickl_file_path, 'rb') as file:
        scaler = pickle.load(file)
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

    return df


def full_pipeline(df: DataFrame) -> DataFrame:
    """Run all stages of feature engineering and get final dataframe
    """
    log.info("Dropping column 'selling_price' ...")
    df.drop('selling_price', axis=1, inplace=True)
    log.info("Changing categorical features ...")
    df = change_cat_features(df=df)
    log.info("Applying simple imputer ...")
    df = fill_empty(df=df, pickl_file_path='source/imputer.pkl')
    log.info("Adding brand class column ...")
    df = add_brand(df=df)
    log.info("Adding car age column ...")
    df = add_car_age(df=df)
    log.info("Adding custom feature ...")
    df = add_custom_feature(df=df)
    log.info("Applying Polynomials ...")
    df = add_polynom(df=df, pickl_file_path='source/polynomials.pkl')
    log.info("Applying one hot encoding ...")
    df = add_ohe(df=df, pickl_file_path='source/ohencoder.pkl')
    log.info("Applying standard scaler...")
    df = add_scaler(df=df, pickl_file_path='source/stscaler.pkl')
    log.info("Final feature dataframe is ready!")

    return df
