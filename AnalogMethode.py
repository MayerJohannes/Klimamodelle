#	Analogmethode
#	David Hinger, Johannes Mayer
#	Klimamodelle SS19
#	version:	2019-05-23
#
# 	data:
#		/home/srvx11/lehre/users/a1127897/JRA-55/mslp_daily rh_daily spfh_daily
#
#	execute:
#		python3 AnalogMethode.py 
#		python3 AnalogMethode.py --ifile1 "mslp_daily/anl_surf125.002_prmsl.195*" --ifile2 "rh_daily/anl_p125.052_rh.195*" --ifile3 "spfh_daily/anl_p125.051_spfh.195*"
import numpy as np
import xarray as xr
import dask
import matplotlib.pylab as plt
#from eofs.xarray import Eof
from scipy.signal import detrend

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='man')

parser.add_argument('--ipath', type=str, help='...', default = '/home/srvx11/lehre/users/a1127897/JRA-55/')
parser.add_argument('--ifile1', type=str, help='...', default = 'mslp_daily/anl_surf125.002_prmsl.1958010100_1958123118.salmi370402')
parser.add_argument('--ifile2', type=str, help='...', default = 'rh_daily/anl_p125.052_rh.1958010100_1958013118.salmi370403')
parser.add_argument('--ifile3', type=str, help='...', default = 'spfh_daily/anl_p125.051_spfh.1958010100_1958013118.salmi370403')

parser.add_argument('--var1', type=str, help='...', default = 'initial_time0_hours')
parser.add_argument('--var2', type=str, help='...', default = 'g0_lat_1')
parser.add_argument('--var3', type=str, help='...', default = 'g0_lon_2')


#parser.add_argument('--arg2', action='store_true', dest='arg2', help='...')
#parser.add_argument('--arg3', type=int, help='...', default = 1)
#parser.add_argument('--arg4', type=float, help='...', default = 1.)
args = parser.parse_args()
ipath = args.ipath
ifile1 = args.ifile1
ifile2 = args.ifile2
ifile3 = args.ifile3
var1 = args.var1
var2 = args.var2
var3 = args.var3

"""
f1 = '*'
f2 = '*'
f3 = '*'
"""
f1 = ''
f2 = ''
f3 = ''
#"""

iData1 = xr.open_mfdataset(ipath+ifile1+f1+'.nc',chunks={var1: 1460})
iData2 = xr.open_mfdataset(ipath+ifile2+f2+'.nc',chunks={var1: 1460})
iData3 = xr.open_mfdataset(ipath+ifile3+f3+'.nc',chunks={var1: 1460})

iData = xr.merge([iData1,iData2,iData3]).chunk({var1: 1460})
iDataVars = list(iData.data_vars)
#print(iDataVars)
print(iData)
iData = iData.rename({var1:'time0'})


iData_DailyMean = iData.resample(time0='1D').mean().chunk({'time0': 730})
print(iData_DailyMean)


iData_DOYMean = iData_DailyMean.rolling(time0=21, center=True).mean().groupby('time0.dayofyear').mean(dim='time0').chunk({'dayofyear': 366})
print(iData_DOYMean)


iData_Anomaly = iData_DailyMean.groupby('time0.dayofyear') - iData_DOYMean
#iData_Anomaly = iData_Anomaly.drop("dayofyear")
print(iData_Anomaly)

## check substraction
#c1 = 450
#c2 = c1 - 365
#print(iData_DailyMean.PRMSL_GDS0_MSL.values[c1,10,10],iData_DOYMean.PRMSL_GDS0_MSL.values[c2,10,10],iData_Anomaly.PRMSL_GDS0_MSL.values[c1,10,10])
#print(iData_DailyMean.RH_GDS0_ISBL.values[c1,10,10],iData_DOYMean.RH_GDS0_ISBL.values[c2,10,10],iData_Anomaly.RH_GDS0_ISBL.values[c1,10,10])
#print(iData_DailyMean.SPFH_GDS0_ISBL.values[c1,10,10],iData_DOYMean.SPFH_GDS0_ISBL.values[c2,10,10],iData_Anomaly.SPFH_GDS0_ISBL.values[c1,10,10])


iData_DOYStd = iData_DailyMean.groupby('time0.dayofyear').std().chunk({'dayofyear': 366})
print(iData_DOYStd)
#print(iData_DOYStd.PRMSL_GDS0_MSL.values[0])
#print(iData_DailyMean.PRMSL_GDS0_MSL.values[0,10,10])


iData_Anomaly_Normalized = iData_Anomaly.groupby('time0.dayofyear')/iData_DOYStd
iData_Anomaly_Normalized = iData_Anomaly_Normalized.chunk({'time0': 730})
print(iData_Anomaly_Normalized)

## check order of magnitude
#print(iData_Anomaly_Normalized.PRMSL_GDS0_MSL.values[100,10,10])
#print(iData_Anomaly_Normalized.RH_GDS0_ISBL.values[100,10,10])
#print(iData_Anomaly_Normalized.SPFH_GDS0_ISBL.values[100,10,10])



