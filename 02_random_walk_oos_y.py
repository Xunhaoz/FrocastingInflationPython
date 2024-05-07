import pandas as pd
import warnings

warnings.filterwarnings('ignore')

n_windows = 180

data = pd.read_csv('data/data.csv', index_col=0)


def roll_prod(ser, width):
    return_ser, ser = ser.copy() + 1, ser.copy() + 1
    for i in range(1, width):
        return_ser *= ser.shift(i)
    return return_ser - 1


y = data[["CPIAUCSL"]]
y['acc3'] = roll_prod(y["CPIAUCSL"], 3)
y['acc6'] = roll_prod(y["CPIAUCSL"], 6)
y['acc12'] = roll_prod(y["CPIAUCSL"], 12)
y_out = y[-n_windows:]
y_out[-180:].to_csv('forecasts/y_out.csv')

for i in range(1, 13):
    y[f't+{i}'] = y["CPIAUCSL"].shift(i)

y['acc3'] = y['acc3'].shift(3)
y['acc6'] = y['acc6'].shift(6)
y['acc12'] = y['acc12'].shift(12)
y[-180:][[f't+{i}' for i in range(1, 13)] + ['acc3', 'acc6', 'acc12']].to_csv('forecasts/rw.csv')
