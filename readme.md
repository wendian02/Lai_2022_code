
Lai, Wendian, Zhongping Lee, Junwei Wang, Yongchao Wang, Rodrigo Garcia, and Huaguo Zhang. A Portable Algorithm to Retrieve Bottom Depth of Optically Shallow Waters from Top-of-Atmosphere Measurements[J]. Journal of Remote Sensing, 2022. https://doi.org/10.34133/2022/9831947.

Apply the NNTOA-B model to Landsat-8 images. 
The depth model and water type model contains in Models folder.

The input is the Landsat-8 Rayleigh-corrected reflectance processed by Acolite(v20210114.0),
The output includes predictions for bottom depth and water type, saved in the nc format in the end.

Because rhorc does not remove cloud pixels, we need to perform additional processing on the original images. We have two different methods for cloud pixel removal.
1. The first method is using the cloud_albedo threshold to remove cloud pixel. (sat_apply_withCloudAlbedo.py)
2. The second method involves distinguishing cloud pixels through a threshold applied to the 865 band.(sat_apply_withoutCloudAlbedo.py)

We recommend using the cloud_albedo method for cloud pixel removal.
Cloud albedo can be acquired by processing through SeaDAS.
