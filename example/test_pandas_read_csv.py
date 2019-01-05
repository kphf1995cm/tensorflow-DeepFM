#env python
# -*- utf-8

import pandas as pd
import numpy as np
import config

def preprocess(df):
	cols = [c for c in df.columns if c not in ["id", "target"]]
	df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
	df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
	return df

def load_data(file_path):
	trainData=pd.read_csv(file_path)
	print(trainData)
	trainData=preprocess(trainData)
	print(trainData)
	cols = [c for c in trainData.columns if c not in ["id", "target"]]
	print(cols)
	cols = [c for c in cols if (not c in config.IGNORE_COLS)]
	print(cols)
	X_train = trainData[cols].values
	y_train = trainData["target"].values
	cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]
	print(X_train)
	print(y_train)
	print(cat_features_indices)

	train_meta=np.zeros((trainData.shape[0],1),dtype=float)
	print(train_meta)

if __name__=='__main__':
	load_data('testdata.csv')




