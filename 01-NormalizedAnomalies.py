#	Computation of normalized anomalies using JRA-55
#	David Hinger, Johannes Mayer
#	Klimamodelle SS19
#	version:	2019-07-23
#
# 	data:
#		/home/srvx11/lehre/users/a1127897/JRA-55/mslp_daily rh_daily spfh_daily
#
#	execute:
#		python3 01-NormalizedAnomalies.py 
#		python3 01-NormalizedAnomalies.py --ifile1 "mslp_daily/anl_surf125.002_prmsl.195*" --ifile2 "rh_daily/anl_p125.052_rh.195*" --ifile3 "spfh_daily/anl_p125.051_spfh.195*"
import numpy as np
import xarray as xr
import dask
import matplotlib.pylab as plt
import iris
from eofs.xarray import Eof
from scipy.signal import detrend
from eofs.multivariate.iris import MultivariateEof

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse

# Data path
#def_path = '/home/srvx11/lehre/users/a1127897/JRA-55/'
def_path = '/home/johannes/klimamodelle/data/'


# parser for input arguments
parser = argparse.ArgumentParser(description='man')
parser.add_argument('--ipath', type=str, help='...', default = def_path)
parser.add_argument('--ifile1', type=str, help='...', default = 'mslp_daily/anl_surf125.002_prmsl.*') #8010100_1958123118.salmi370402')
parser.add_argument('--ifile2', type=str, help='...', default = 'rh_daily/anl_p125.052_rh.*') #8010100_1958013118.salmi370403')
parser.add_argument('--ifile3', type=str, help='...', default = 'spfh_daily/anl_p125.051_spfh.*') #8010100_1958013118.salmi370403')

parser.add_argument('--var1', type=str, help='...', default = 'initial_time0_hours')
parser.add_argument('--var2', type=str, help='...', default = 'g0_lat_1')
parser.add_argument('--var3', type=str, help='...', default = 'g0_lon_2')

args = parser.parse_args()
ipath = args.ipath
ifile1 = args.ifile1
ifile2 = args.ifile2
ifile3 = args.ifile3
var1 = args.var1
var2 = args.var2
var3 = args.var3

# Read sea level pressure, rel. and spec. humidity
iData1 = xr.open_mfdataset(ipath+ifile1+'.nc',chunks={var1: 1460})
iData2 = xr.open_mfdataset(ipath+ifile2+'.nc',chunks={var1: 1460})
iData3 = xr.open_mfdataset(ipath+ifile3+'.nc',chunks={var1: 1460})

# Merge xarray datasets
iData = xr.merge([iData1,iData2,iData3]).chunk({var1: 1460})

iDataVars = list(iData.data_vars)
iData = iData.rename({var1:'time0'})

# Roll lon coordinate such that lon = [-180..180]
iData.coords['g0_lon_2'] = (iData.coords['g0_lon_2'] + 180) % 360 - 180
print(iData)

# Compute daily means
iData_DailyMean = iData.resample(time0='1D').mean().chunk({'time0': 730})
print(iData_DailyMean)

# Create rolling window
iData_roll = iData_DailyMean.rolling(time0=21, center=True).construct('window_dim')

# Compute Day of year (DOY) means
iData_DOYMean = iData_roll.groupby('time0.dayofyear').mean(dim=['window_dim','time0']).chunk({'dayofyear': 366})
print(iData_DOYMean)

# Compute DOY anomalies by substracting DOY means
iData_Anomaly = iData_DailyMean.groupby('time0.dayofyear') - iData_DOYMean
print(iData_Anomaly)

## check substraction
#c1 = 450
#c2 = c1 - 365
#print(iData_DailyMean.PRMSL_GDS0_MSL.values[c1,10,10],iData_DOYMean.PRMSL_GDS0_MSL.values[c2,10,10],iData_Anomaly.PRMSL_GDS0_MSL.values[c1,10,10])
#print(iData_DailyMean.RH_GDS0_ISBL.values[c1,10,10],iData_DOYMean.RH_GDS0_ISBL.values[c2,10,10],iData_Anomaly.RH_GDS0_ISBL.values[c1,10,10])
#print(iData_DailyMean.SPFH_GDS0_ISBL.values[c1,10,10],iData_DOYMean.SPFH_GDS0_ISBL.values[c2,10,10],iData_Anomaly.SPFH_GDS0_ISBL.values[c1,10,10])

# Compute DOY standard deviation
iData_DOYStd = iData_roll.groupby('time0.dayofyear').std().chunk({'dayofyear': 366})
print(iData_DOYStd)

# Compute normalized anomalies
iData_Anomaly_Normalized = iData_Anomaly.groupby('time0.dayofyear')/iData_DOYStd
iData_Anomaly_Normalized = iData_Anomaly_Normalized.chunk({'time0': 730})
print(iData_Anomaly_Normalized)

## check order of magnitude
#print(iData_Anomaly_Normalized.PRMSL_GDS0_MSL.values[100,10,10])
#print(iData_Anomaly_Normalized.RH_GDS0_ISBL.values[100,10,10])
#print(iData_Anomaly_Normalized.SPFH_GDS0_ISBL.values[100,10,10])

# Drop redundant time dimension
iData_Anomaly_Normalized = iData_Anomaly_Normalized.drop('initial_time0_encoded')

# Save normalized anomalies to file
iData_Anomaly_Normalized.to_netcdf('anr.nc',mode='w')



