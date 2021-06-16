import pandas as pd
from common import read_df, filter_df

dataroot = './data'

num = 4
def prep(num):
	X_train, X_test = read_df(dataroot, num)

	dummies = ['weekday', 'hour']
	for col in dummies:
		X_train = pd.concat([X_train, pd.get_dummies(X_train[col], prefix=col)], axis=1)






