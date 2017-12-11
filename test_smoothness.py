import pandas as pd
import datetime as dt

from smoothness_estimation import *


def date_parser(integer):
    return dt.datetime.strptime(str(integer), "%Y%m%d")

# test of smoothness estimation on realised variance data of SP500
df = pd.read_csv('sp.csv', sep=';')
df = df.loc[:3722, :]
df = df.dropna()
df.loc[:, "DateID"] = df.loc[:, "DateID"].apply(date_parser)
df = df.set_index("DateID")
# new_index = pd.date_range(df.index[0], df.index[-1])
# df = df.reindex(new_index)
# df = df.interpolate()
varProcess = df.iloc[:, 0].values
# t = (df.index[-1] - df.index[0]).days
t = len(varProcess)
plotVar(varProcess, t)
