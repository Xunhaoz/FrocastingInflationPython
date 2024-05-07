from pathlib import Path
import pandas as pd
import numpy as np

forecasts = Path('forecasts').glob('*.csv')
y_out = pd.read_csv('forecasts/y_out.csv', index_col=0)
rw = pd.read_csv('forecasts/rw.csv', index_col=0)

mse_payload = {}
mae_payload = {}
for forecast in forecasts:
    if forecast.stem == 'y_out':
        continue

    df = pd.read_csv(forecast, index_col=0)

    mse_payload[forecast.stem] = np.concatenate([
        np.average(np.square(df.values[:, :12] - y_out.values[:, :1]), axis=0),
        np.average(np.square(df.values[2:, 12] - y_out.values[2:, 1]), axis=0, keepdims=True),
        np.average(np.square(df.values[5:, 13] - y_out.values[5:, 2]), axis=0, keepdims=True),
        np.average(np.square(df.values[12:, 14] - y_out.values[12:, 3]), axis=0, keepdims=True)
    ])

    mae_payload[forecast.stem] = np.concatenate([
        np.average(np.abs(df.values[:, :12] - y_out.values[:, :1]), axis=0),
        np.average(np.abs(df.values[2:, 12] - y_out.values[2:, 1]), axis=0, keepdims=True),
        np.average(np.abs(df.values[5:, 13] - y_out.values[5:, 2]), axis=0, keepdims=True),
        np.average(np.abs(df.values[12:, 14] - y_out.values[12:, 3]), axis=0, keepdims=True)
    ])


mse_df = pd.DataFrame(
    mse_payload,
    index=rw.columns,
).T

mae_df = pd.DataFrame(
    mae_payload,
    index=rw.columns,
).T

(mse_df / mse_df.loc['rw']).to_csv('mse.csv')
(mae_df / mae_df.loc['rw']).to_csv('mae.csv')