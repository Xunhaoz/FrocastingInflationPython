import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

start_date = "1960-01-01"
end_date = "2015-12-01"


def preprocess_data(df, transform=False):
    transform_code, df = df.iloc[0], df[1:]
    df.index = pd.to_datetime(df.index)

    if transform:
        for id_, c_name in enumerate(df.columns):

            if transform_code[c_name] == 1:
                continue
            elif transform_code[c_name] == 2:
                df[c_name] = df[c_name] - df[c_name].shift(1)
            elif transform_code[c_name] == 3:
                df[c_name] = df[c_name] - df[c_name].shift(1)
                df[c_name] = df[c_name] - df[c_name].shift(1)
            elif transform_code[c_name] == 4:
                df[c_name] = np.log(df[c_name])
            elif transform_code[c_name] == 5:
                df[c_name] = np.log(df[c_name])
                df[c_name] = df[c_name] - df[c_name].shift(1)
            elif transform_code[c_name] == 6:
                if 118 > id_ >= 98:
                    df[c_name] = np.log(df[c_name])
                    df[c_name] = df[c_name] - df[c_name].shift(1)
                    df[c_name] = df[c_name] * 100
                else:
                    df[c_name] = np.log(df[c_name])
                    df[c_name] = df[c_name] - df[c_name].shift(1)
                    df[c_name] = df[c_name] - df[c_name].shift(1)
            elif transform_code[c_name] == 7:
                df[c_name] = df[c_name] / df[c_name].shift(1) - 1

    return df


data = pd.read_csv(
    'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv', index_col=0
)
data = preprocess_data(data, transform=True)[start_date:end_date].dropna(axis=1)

data_raw = pd.read_csv(
    'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv', index_col=0
)
data_raw = preprocess_data(data_raw, transform=False)[start_date:end_date]

data.to_csv("data/data.csv")
