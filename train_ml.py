import os
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

# from dask.distributed import Client
# from dask_cuda import LocalCUDACluster
# import dask
# import cudf
# import dask_cudf

# GPUs = ','.join([str(i) for i in range(0,4)])
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUs


year = 2018

# load crop samples
def load_crop_samples(year):
    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
    crop_samples = f'{parent_dir}/data/S2_{year}_crop_samples.nc'
    ds = xr.open_dataset(crop_samples)

    lucode = ds['lucode'].values

    band_values = ds['band_values'].values.reshape(-1, lucode.size)

    # lucodes = ['AC00', 'AC01', 'AC03', 'AC04', 'AC05', 'AC06', 'AC07', 'AC09',
    #             'AC10', 'AC100', 'AC14', 'AC15', 'AC16', 'AC17', 'AC18', 'AC19',
    #             'AC20', 'AC22', 'AC23', 'AC24', 'AC26', 'AC27', 'AC30', 'AC32',
    #             'AC34', 'AC35', 'AC36', 'AC37', 'AC38', 'AC41', 'AC44', 'AC45',
    #             'AC50', 'AC52', 'AC58', 'AC59', 'AC60', 'AC61', 'AC62', 'AC63',
    #             'AC64', 'AC65', 'AC66', 'AC67', 'AC68', 'AC69', 'AC70', 'AC71',
    #             'AC72', 'AC74', 'AC81', 'AC88', 'AC90', 'AC92', 'AC94', 'CA02',
    #             'FA01', 'HE02', 'HEAT', 'LG01', 'LG02', 'LG03', 'LG04', 'LG06',
    #             'LG07', 'LG08', 'LG09', 'LG11', 'LG13', 'LG14', 'LG15', 'LG16',
    #             'LG20', 'LG21', 'NA01', 'NU01', 'PG01', 'SR01', 'TC01', 'WA00',
    #             'WA01', 'WO12']

    # ulucode = np.unique(lucode)

    # # check if all the lucodes are in the list
    # lucode_not_in = [i for i in ulucode if i not in lucodes]
    # if lucode_not_in:
    #     raise ValueError(f'Lucodes {lucode_not_in} not in the list of lucodes!')

    # # lucode_index = np.array([lucodes.index(l) for l in lucode])
    # # Convert lucodes to a dictionary that maps each code to its index for O(1) lookups
    # lucodes_dict = {code: idx for idx, code in enumerate(lucodes)}

    # # Use the dictionary to find the index of each value in lucode
    # try:
    #     lucode_index = np.array([lucodes_dict[l] for l in lucode])
    # except KeyError as e:
    #     raise ValueError(f"Lucode {e.args[0]} not in the list of lucodes!")
    # y = lucode_index


    le = LabelEncoder()
    y = le.fit_transform(lucode)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    # column names is the combination of band names and month
    band_name = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    month = np.arange(1, 13)

    column_names = [f'{b}_{m}' for b in band_name for m in month]
    df = pd.DataFrame(band_values.T, columns=column_names)
    df['lucode'] = y

    return df, le_name_mapping

data, le_name_mapping = load_crop_samples(year)


from sklearn.model_selection import train_test_split

# Separate target variable
X = data.drop('lucode', axis=1)
y = data['lucode']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from xgboost import XGBClassifier

# Create an instance of the XGBClassifier
# model = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=42)


model = XGBClassifier(
                    learning_rate =0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    # objective= 'binary:logistic',
                    # nthread=4,
                    # scale_pos_weight=1,
                    seed=27,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=3,
                    )


# Fit the model to the training data
model.fit(X_train, y_train, verbose=True)

# compute the accuracy of the model
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# compute the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# compute the classification report
from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)
print(class_report)
# save the accuracy, confusion matrix, and classification report
with open(f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_metrics_v2.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{class_report}\n")

import os
model_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_model.json'
le_name_mapping_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/lucode_mapping_{year}.csv'
if os.path.exists(model_path):
    os.remove(model_path)
if os.path.exists(le_name_mapping_path):
    os.remove(le_name_mapping_path)

# save the model
model.save_model(model_path)
le_name_mapping = pd.DataFrame.from_dict(le_name_mapping, orient='index', columns=['lucode'])
le_name_mapping.to_csv(le_name_mapping_path)


# def get_cluster():
#     cluster = LocalCUDACluster()
#     client = Client(cluster)
#     return client



# client = get_cluster()



# # convert to cudf
# gdf = cudf.DataFrame.from_pandas(df)

# df = None
# band_values = None


# # convert to dask_cudf
# dgdf = dask_cudf.from_cudf(gdf)
# gdf = None

# dtrain = xgb.dask.DaskQuantileDMatrix(client, dgdf[column_names], dgdf['lucode'])
# print(dtrain)


# FOLDS = 5
# SEED = 42

# LR = 0.1

# xgb_parms = { 
#     'max_depth':4, 
#     'learning_rate':LR, 
#     'subsample':0.7,
#     'colsample_bytree':0.5, 
#     'eval_metric':'map',
#     'objective':'multi:softmax',
#     'scale_pos_weight':8,
#     'tree_method':'gpu_hist',
#     'predictor':'gpu_predictor',
#     'random_state':SEED
# }

# output = xgb.dask.train(
#         client,
#         xgb_parms,
#         dtrain,
#         num_boost_round=100,
#         evals=[(dtrain, "train")],
#     )
# # save the model
# model = output['booster']
# model.save_model(f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_model.json')
