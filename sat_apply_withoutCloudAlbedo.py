"""
Lai, Wendian, Zhongping Lee, Junwei Wang, Yongchao Wang, Rodrigo Garcia, and Huaguo Zhang. A Portable Algorithm to Retrieve Bottom Depth of Optically Shallow Waters from Top-of-Atmosphere Measurements[J]. Journal of Remote Sensing, 2022. https://doi.org/10.34133/2022/9831947.

Apply the NNTOA-B model to Landsat-8 images. The depth model and water type model contains in Models folder.

The input is the Landsat-8 Rayleigh-corrected reflectance processed by Acolite(v20210114.0),
The output includes predictions for bottom depth and water type, saved in the nc format in the end.

Because rhorc does not remove cloud pixels, we need to perform additional processing on the original images.
We have two different methods for cloud pixel removal.
The first method is using the cloud_albedo threshold to remove cloud pixel. (sat_apply_withCloudAlbedo.py)
The second method involves distinguishing cloud pixels through a threshold applied to the 865 band.(sat_apply_withoutCloudAlbedo.py)
We recommend using the cloud_albedo method for cloud pixel removal.
Cloud albedo can be acquired by processing through SeaDAS.

For any inquiries, please contact wendian_lai@163.com.
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import netCDF4 as nc
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import os

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

# read nc file
fn = 'L8_OLI_2020_03_21_02_53_54_122049_L2W.nc'  # rhorc nc file output from Acolite(v20210114.0)
f = nc.Dataset(fn)

# # read data
lat = np.array(f['lat'])
lon = np.array(f['lon'])
#
rhorc_443 = np.array(f['rhorc_443'])
rhorc_482 = np.array(f['rhorc_483'])
rhorc_561 = np.array(f['rhorc_561'])
rhorc_655 = np.array(f['rhorc_655'])
rhorc_865 = np.array(f['rhorc_865'])
rhorc_1609 = np.array(f['rhorc_1609'])
rhorc_2201 = np.array(f['rhorc_2201'])

# cloud_albedo = np.array(f_cloud['cloud_albedo'])
# cloud_albedo = np.array(f_cloud['geophysical_data']['cloud_albedo'])
(n_row, n_col) = np.shape(rhorc_443)

# print('shape of image :', np.shape(Rrs443))


# read mat file
# matData = sio.loadmat(r'E:\Rrs_LAT_LON.mat')
# lon = matData['LON']
# lat = matData['LAT']
# Rrs443 = np.array(matData['Rrs443'])
# Rrs482 = np.array(matData['Rrs482'])
# Rrs561 = np.array(matData['Rrs561'])
# Rrs655 = np.array(matData['Rrs655'])
# (n_row, n_col) = np.shape(Rrs443)
# print('shape of image :', np.shape(Rrs443))

# input data
input1 = rhorc_443.flatten()[:, np.newaxis]
input2 = rhorc_482.flatten()[:, np.newaxis]
input3 = rhorc_561.flatten()[:, np.newaxis]
input4 = rhorc_655.flatten()[:, np.newaxis]
input5 = rhorc_865.flatten()[:, np.newaxis]
input6 = rhorc_1609.flatten()[:, np.newaxis]
input7 = rhorc_2201.flatten()[:, np.newaxis]
# cloud_albedo = cloud_albedo.flatten()[:, np.newaxis]

X = np.concatenate([input1, input2, input3, input4, input5, input6, input7], axis=1)

x_chs = pd.DataFrame(X)
# cloud_albedo = pd.DataFrame(cloud_albedo)
# quality control
x_chs[x_chs < 0] = np.nan
x_chs[x_chs > 1] = np.nan

# choose the band 865 as the cloud threshold
cloud_thre = pd.DataFrame(input5)
# delete cloud pixel
x_chs.iloc[cloud_thre.iloc[:, 0] > 0.03, :] = np.nan  # cloud threshold of L8 (manually selected After observing cloud pixels in the rhorc(865))

x_chs['idx'] = list(range(0, np.size(X[:, 0])))  # flag location of each pixel
x_chs = x_chs.dropna(axis=0, how='any')  # drop nan Rrs

X_chs = x_chs.copy()
X_chs.pop('idx')

depth_model = load_model('./Models/depth_model.h5', custom_objects={'r2_keras': r2_keras})
class_model = load_model('./Models/classify_model_global.h5')

X_norm_h = X_chs
y_predict = depth_model.predict(X_norm_h)
y_predict[y_predict < 0] = np.nan
y_predict[np.isinf(y_predict)] = np.nan

#  reshape output
raw = pd.DataFrame(np.hstack([y_predict, x_chs[['idx']]]), columns=['predict', 'idx'])
predict_1 = pd.DataFrame(data=np.arange(0, np.size(X[:, 1])), columns=['idx'])
predict = pd.merge(predict_1, raw, how='outer', on='idx')

# reshape output to image size
depth = np.array(predict['predict']).reshape(n_row, n_col)

# predict class
X_norm_class = X_chs
y_predict = class_model.predict(X_norm_class)
y_predict[y_predict < 0] = np.nan
y_predict[np.isinf(y_predict)] = np.nan

#  reshape output
raw = pd.DataFrame(np.hstack([y_predict, x_chs[['idx']]]), columns=['predict', 'idx'])
predict_1 = pd.DataFrame(data=np.arange(0, np.size(X[:, 1])), columns=['idx'])
predict = pd.merge(predict_1, raw, how='outer', on='idx')

depth_class = np.array(predict['predict']).reshape(n_row, n_col)  # reshape output to image size

# save results, and plot result map with Matlab m_map
# sio.savemat('Depth_class_Mask.mat',
#             mdict={'lon': lon, 'lat': lat, 'h': output})

# save H and water type into NC file
f_w = nc.Dataset(r'rs_122049_20200321.nc', 'w', format='NETCDF4')
# define dimensions
f_w.createDimension('x', size=lon.shape[1])
f_w.createDimension('y', size=lon.shape[0])

# create variables
lat_w = f_w.createVariable('lat', np.float32, ('y', 'x'))
lon_w = f_w.createVariable('lon', np.float32, ('y', 'x'))
H_w = f_w.createVariable('H', np.float32, ('y', 'x'))
class_w = f_w.createVariable('class', np.float32, ('y', 'x'))

# lon/lat/output are np.array, 1d, 1d, 2d
lon_w[:] = lon
lat_w[:] = lat

class_w[:] = depth_class  # the 0 refer the optically shallow water, and the 1 refer the optically deep water
depth[depth_class == 1] = np.nan  # mask the optically deep water
H_w[:] = depth

f_w.close()
