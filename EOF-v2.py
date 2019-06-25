# execute:
#	mpiexec -n 4 python3 EOF-v2.py

import numpy as np
import xarray as xr
import dask
import matplotlib.pylab as plt
import iris
from eofs.xarray import Eof
from scipy.signal import detrend
from eofs.multivariate.iris import MultivariateEof
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse
import csv

def RMSE(field1,field2):
	rmse = np.sqrt(((field1 - field2)**2.).mean())
	return rmse

#from mpi4py import MPI

#def_path = '/home/srvx11/lehre/users/a1127897/JRA-55/'
#def_path = '/home/johannes/klimamodelle/data/'

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

#print("Number of processes:", size)

#exit()

var1 = 'initial_time0_hours'

#iData1 = xr.open_mfdataset('data/mslp_daily/anl_surf125.002_prmsl.1959*.nc',chunks={var1: 1460})
#iData2 = xr.open_mfdataset('data/rh_daily/anl_p125.052_rh.1959*.nc',chunks={var1: 1460})
#iData3 = xr.open_mfdataset('data/spfh_daily/anl_p125.051_spfh.1959*.nc',chunks={var1: 1460})
#iData = xr.merge([iData1,iData2,iData3]).chunk({var1: 1460}).drop('initial_time0_encoded')
#td = iData.sel(initial_time0_hours=slice('1959-01-01'))
#td = td.mean(dim=['initial_time0_hours'])
#td.coords['g0_lon_2'] = (td.coords['g0_lon_2'] + 180) % 360 - 180
#iris_spfh_td = td.SPFH_GDS0_ISBL.to_iris()
#iris_prmsl_td = td.PRMSL_GDS0_MSL.to_iris()
#iris_rh_td = td.RH_GDS0_ISBL.to_iris()
#print(iris_spfh_td)
#exit()



#ifile = '/mnt/sdb2/data/anr.nc'
ifile = '/home/johannes/klimamodelle/anr.nc'
syear = 1958
nyears = 61

new_data_array = {}
drop_list = []
rmse_spfh = []
rmse_prmsl = []
rmse_rh = []


iData_Anomaly_Normalized = xr.open_dataset(ifile)
ntime = iData_Anomaly_Normalized.time0.values.shape[0]

#plt.figure()
#plt.contourf(iData_Anomaly_Normalized.sel(time0=slice('1984-01-02','1984-01-02')).SPFH_GDS0_ISBL.values[0,:,:],vmin=-2.,vmax=2.)
#plt.colorbar()

#plt.figure()
#plt.contourf(iData_Anomaly_Normalized.sel(time0=slice('2004-12-23','2004-12-23')).SPFH_GDS0_ISBL.values[0,:,:],vmin=-2.,vmax=2.)
#plt.colorbar()
#plt.show()
#exit()


#for i in range(61):
#	if (syear+i)%4 == 0:
#		drop_list += [np.datetime64(str(syear+i)+'-02-29')]
#print(drop_list)
#iData_Anomaly_Normalized = iData_Anomaly_Normalized.drop(drop_list, dim='time0')
#iData_Anomaly_Normalized = iData_Anomaly_Normalized.sel(time0=slice('1958-01-01','1969-12-31'))

iData_anom_norm_roll = iData_Anomaly_Normalized.rolling(time0=21, center=True).construct('window_dim')

cnt = 0
for _,ds_test in iData_anom_norm_roll.groupby("time0.dayofyear"):
	cnt = cnt + 1 # 1...366

	if cnt > 0 and cnt < 400:
		date_str = ds_test.time0.values	# for target day
		
		ds_test = ds_test.rename({"time0":"time_old"}).stack(time=('time_old', 'window_dim')).transpose('time', 'g0_lat_1', 'g0_lon_2').dropna('time')
		ds_test.coords['time'].attrs['axis'] = 'T'
		iris_spfh = ds_test.SPFH_GDS0_ISBL.assign_coords(time=range(0,len(ds_test.time))).to_iris()
		iris_prmsl = ds_test.PRMSL_GDS0_MSL.assign_coords(time=range(0,len(ds_test.time))).to_iris()
		iris_rh = ds_test.RH_GDS0_ISBL.assign_coords(time=range(0,len(ds_test.time))).to_iris()

		msolver = MultivariateEof([iris_prmsl,iris_rh,iris_spfh],center=False)

		#ieof = msolver.eofs(neofs=5) #weights="coslat")
		#eof = ieof[1]

		date_arr = ds_test.time.values

		neigs = 1
		vari = 0.0
		while vari < 0.9:
			vari = sum(msolver.varianceFraction(neigs=neigs).data)
			neigs += 1
			#print(cnt_vari, vari)

		print(vari)

		pc = msolver.pcs(npcs=neigs)		
		#exit()

	#	if cnt == 60:
	#		print(date_arr.shape)
	#		print(date_str.shape)
	#		exit()

		print('cnt = ',cnt,', neigs = ',neigs)#,date_str)
		if cnt == 366: nyears = 15
		for i in range(nyears):
			td_str = str(date_str[i])[:10]
			#print(td_str)
			td = iData_Anomaly_Normalized.sel(time0=slice(td_str,td_str))
			iris_spfh_td = td.SPFH_GDS0_ISBL.to_iris()
			iris_prmsl_td = td.PRMSL_GDS0_MSL.to_iris()
			iris_rh_td = td.RH_GDS0_ISBL.to_iris()

			ppc = msolver.projectField([iris_prmsl_td,iris_rh_td,iris_spfh_td],neofs=neigs)

			norm = []
			for j in range(pc.shape[0]):
				if td_str == str(date_arr[j][0])[:10]:
					#print(td_str,str(date_all_str[i])[:10])
					norm += [999999.]
				else:
					norm += [np.sum((pc.data[j] - ppc.data[0])**2.0)]

			min_index = np.argmin(norm)	
			newday_str = str(date_arr[min_index][0]+datetime.timedelta(days=date_arr[min_index][1]-10))[:10]
			print(i,':: ',td_str,':', newday_str)  
			new_data_array[td_str] = newday_str 

			#if cnt == 61:  exit() #break
			td_field = iData_Anomaly_Normalized.sel(time0=slice(td_str,td_str))
			nd_field = iData_Anomaly_Normalized.sel(time0=slice(newday_str,newday_str))

			rmse_spfh += [RMSE(td_field.SPFH_GDS0_ISBL.values[0,:,:],nd_field.SPFH_GDS0_ISBL.values[0,:,:])]
			rmse_prmsl += [RMSE(td_field.PRMSL_GDS0_MSL.values[0,:,:],nd_field.PRMSL_GDS0_MSL.values[0,:,:])]
			rmse_rh += [RMSE(td_field.RH_GDS0_ISBL.values[0,:,:],nd_field.RH_GDS0_ISBL.values[0,:,:])]
			#print(rmse_spfh[-1],rmse_prmsl[-1],rmse_rh[-1])
			#if i == 4: break
	#break

#
#print(rmse_list.shape)
#exit()

w = csv.writer(open("datelist-2.csv", "w"))
i = 0
w.writerow(["target_day","new_day","rmse_spfh","rmse_prmsl","rmse_rh"])
for key, val in new_data_array.items():
	w.writerow([key, val, rmse_spfh[i],rmse_prmsl[i], rmse_rh[i]])
	i += 1

