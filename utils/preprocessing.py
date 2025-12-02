import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessing(
    df: pd.DataFrame, years: list, countries: list, cols: list, features: list
) -> pd.DataFrame:
    """select data and features to process and log-transform floats"""
    # dfl = df.copy()
    df = df[df["year"].isin(years)]
    df = df[df["country"].isin(countries)]
    # dfl = dfl[cols + features]
    df.reset_index(drop=True, inplace=True)

    # columns to log-transform (floats) :
    log_features = df[features].select_dtypes(include=["float64"]).columns.tolist()
    # floats = dfl.dtypes[df.dtypes == 'float64'].index.tolist() # same result

    # log transform because asyetrical distributions
    df_log = df.copy()
    for feat in log_features:
        df_log[feat] = np.log1p(df[feat])  # log1p to handle zero values safely

    # standardisation :
    scaler = StandardScaler()
    X = scaler.fit_transform(df_log[log_features])

    return df, X


def test_dataset(dataset):
    """utility to test if countries have a valid iso_code"""
    cty_list = ["Brazil", "Russia", "India", "nocountry"]
    # cty_list = ["Somalia", "Sudan", "Greenland", "Eritrea", "Guyana", "Suriname", "South Sudan", "test", "Papua New Guinea", "Solomon Islands", "Fiji", "Tuvalu"]
    for cty in cty_list:
        try:
            print(
                f"{dataset[dataset["country"].str.contains(cty)]["iso_code"].unique()[0]}: {cty}"
            )
        except IndexError:
            print(f"! MISSING ! {cty}")
