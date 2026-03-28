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

'''
AC01	Spring Barley
AC03	Beet
AC04	Borage
AC05	Buckwheat
AC06	Canary Seed
AC07	Carrot
AC09	Chicory
AC10	Daffodil
AC14	Hemp
AC15	Lettuce
AC16	Spring Linseed 
AC17	Maize
AC18	Millet
AC19	Spring Oats
AC20	Onions
AC22	Parsley
AC23	Parsnips
AC24	Spring Rye
AC26	Spinach
AC27	Strawberry
AC30	Spring Triticale
AC32	Spring Wheat
AC34	Spring Cabbage
AC35	Turnip
AC36	Spring Oilseed
AC37	Brown Mustard
AC38	Mustard
AC41	Radish
AC44	Potato
AC45	Tomato
AC50	Squash
AC52	Siam Pumpkin
AC58	Mixed Crop-Group 1
AC59	Mixed Crop-Group 2
AC60	Mixed Crop-Group 3
AC61	Mixed Crop-Group 4
AC62	Mixed Crop-Group 5
AC63	Winter Barley
AC64	Winter Linseed
AC65	Winter Oats
AC66	Winter Wheat
AC67	Winter Oilseed
AC68	Winter Rye
AC69	Winter Triticale
AC70	Winter Cabbage
AC71	Coriander
AC72	Corn gromwell
AC74	Phacelia
AC81	Poppy
AC88	Sunflower
AC90	Gladioli
AC92	Sorghum
AC94	Sweet William
AC100	Italian Ryegrass
CA02	Cover Crop
LG01	Chickpea
LG02	Fenugreek
LG03	Spring Field beans
LG04	Green Beans
LG06	Lupins
LG07	Spring Peas
LG09	Cowpea
LG08	Soya
LG11	Lucerne
LG13	Sainfoin
LG14	Clover
LG15	Mixed Crops–Group 1 Leguminous
LG16	Mixed Crops–Group 2 Leguminous
LG20	Winter Field beans
LG21	Winter Peas
SR01	Short Rotation Coppice
FA01	Fallow Land
HE02	Heathland and Bracken
HEAT	Heather
PG01	Grass
NA01	Non-vegetated or sparsely-vegetated Land
WA00	Water
TC01	Perennial Crops and Isolated Trees
NU01	Nursery Crops
WO12	Trees and Scrubs, short Woody plants, hedgerows
AC00	Unknown or Mixed Vegetation
'''


year = 2018

# load crop samples
def load_crop_samples(year):
    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
    crop_samples = f'{parent_dir}/data/S2_{year}_crop_samples.nc'
    ds = xr.open_dataset(crop_samples)

    lucode = ds['lucode'].values

    band_values = ds['band_values'].values.reshape(-1, lucode.size)

    # # select first 1000 samples for testing
    # n_samples = 1000000
    # lucode = lucode[:n_samples]
    # band_values = band_values[:, :n_samples]


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


    # le = LabelEncoder()
    # y = le.fit_transform(lucode)
    
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)
    # only need to get AC66 for now
    y = lucode == 'AC66'
    le_name_mapping = {'AC66': 1, 'Other': 0}
    
    # column names is the combination of band names and month
    band_name = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    month = np.arange(1, 13)

    column_names = [f'{b}_{m}' for b in band_name for m in month]
    df = pd.DataFrame(band_values.T, columns=column_names)
    df['lucode'] = y.astype(int)

    return df, le_name_mapping

data, le_name_mapping = load_crop_samples(year)


from sklearn.model_selection import train_test_split
import cudf
# import dask_cudf

# Separate target variable
X = data.drop('lucode', axis=1)
y = data['lucode']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    X_train = cudf.DataFrame.from_pandas(X_train)
    # X_test = cudf.DataFrame.from_pandas(X_test)
    y_train = cudf.Series.from_pandas(y_train)
    # y_test = cudf.Series.from_pandas(y_test)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Leave most parameters as default
    model = xgb.XGBClassifier(device="cuda",
                        learning_rate = 0.01,
                        n_estimators=5000,
                        max_depth=15,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        early_stopping_rounds=10,
                        # objective= 'binary:logistic',
                        # nthread=4,
                        # scale_pos_weight=1,
                        seed=27,
                        n_jobs=-1,
                        tree_method="hist",
                        verbosity=3,
                        )

    import time
    # Train model
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    gpu_res = model.evals_result()
    print("GPU Training Time: %s seconds" % (str(time.time() - start)))


    import os
    model_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_model_winter_wheat.json'
    le_name_mapping_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/lucode_mapping_{year}_winter_wheat.csv'
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(le_name_mapping_path):
        os.remove(le_name_mapping_path)

    # save the model
    model.save_model(model_path)

    # le_name_mapping = pd.DataFrame.from_dict(le_name_mapping, orient='index', columns=['lucode'])
    # le_name_mapping.to_csv(le_name_mapping_path)

    # delete the training data to free up cuda memory
    del X_train
    del y_train

    return model

model = train_model(X_train, X_test, y_train, y_test)

# compute the accuracy of the model
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# compute the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix.tolist())
# compute the classification report
from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)
print(class_report)
# save the accuracy, confusion matrix, and classification report
with open(f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_metrics_cudf.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{class_report}\n")

# import os
# model_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_model.json'
# le_name_mapping_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/lucode_mapping_{year}.csv'
# if os.path.exists(model_path):
#     os.remove(model_path)
# if os.path.exists(le_name_mapping_path):
#     os.remove(le_name_mapping_path)

# # save the model
# model.save_model(model_path)
# le_name_mapping = pd.DataFrame.from_dict(le_name_mapping, orient='index', columns=['lucode'])
# le_name_mapping.to_csv(le_name_mapping_path)

