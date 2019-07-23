#	Computation of PCs and PPCs using normalized anomalies 
#	David Hinger, Johannes Mayer
#	Klimamodelle SS19
#	version:	2019-07-23
#
# 	data:
#		anr.nc (from computation of normalized anomalies)
#
#	execute:
#		python3 02-EOF.py

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

# RMSE definition
def RMSE(field1,field2):
	rmse = np.sqrt(((field1 - field2)**2.).mean())
	return rmse

# time variable name
var1 = 'initial_time0_hours'

# path of file including normalized anomalies
ifile = '/home/johannes/klimamodelle/anr.nc'
syear = 1958
nyears = 61

new_data_array = {}
drop_list = []
rmse_spfh = []
rmse_prmsl = []
rmse_rh = []

# Read normalized anomalies
iData_Anomaly_Normalized = xr.open_dataset(ifile)
ntime = iData_Anomaly_Normalized.time0.values.shape[0]
iData_anom_norm_roll = iData_Anomaly_Normalized.rolling(time0=21, center=True).construct('window_dim')

# Output file
file1 = csv.writer(open("datelist-v4.csv","w"))

cnt = 0
for _,ds_test in iData_anom_norm_roll.groupby("time0.dayofyear"):
	cnt = cnt + 1 # 1...366

	# if condition for debugging, redundant
	if cnt > 0 and cnt < 400:
		date_str = ds_test.time0.values	# for target day
		ds_test = ds_test.rename({"time0":"time_old"}).stack(time=('time_old', 'window_dim')).transpose('time', 'g0_lat_1', 'g0_lon_2').dropna('time')
		ds_test.coords['time'].attrs['axis'] = 'T'

		# Transform normalized anomaly field to iris cube (for multivariate EOF)
		iris_spfh = ds_test.SPFH_GDS0_ISBL.assign_coords(time=range(0,len(ds_test.time))).to_iris()
		iris_prmsl = ds_test.PRMSL_GDS0_MSL.assign_coords(time=range(0,len(ds_test.time))).to_iris()
		iris_rh = ds_test.RH_GDS0_ISBL.assign_coords(time=range(0,len(ds_test.time))).to_iris()

		# Allocate multivariate EOF solver
		msolver = MultivariateEof([iris_prmsl,iris_rh,iris_spfh],center=False)



## EOF PLOT
#		ieof = msolver.eofs(neofs=4) 
#		eof = ieof[0]
#		plt.figure()
#		for i in range(4):
#			plt.subplot(2,2,i+1)
#			plt.xlabel("$^\circ$E")
#			plt.ylabel("$^\circ$N")
#			plt.title("EOF"+str(i+1))
#			cb = plt.pcolormesh(np.linspace(-10.,25.,29),np.linspace(32.5,67.5,29),eof[i,:,:].data,vmin=-0.03,vmax=0.03)
#			plt.colorbar(cb)
#		plt.show()
#		exit()

		# Date array of current DOY time series
		date_arr = ds_test.time.values

		# Compute number of EOFs explaining 90 perc. of variance
		neigs = 1
		vari = 0.0
		while vari < 0.9:
			vari = sum(msolver.varianceFraction(neigs=neigs).data)
			neigs += 1

		# Compute principle components of current DOY time series
		pc = msolver.pcs(npcs=neigs)

## PC PLOT
#		plt.figure()
#		for i in range(4):
#			plt.subplot(4,1,i+1)
#			txt = "PC(EOF{:d})".format(i+1)
#			plt.ylabel(txt)
#			plt.xlim(0,1271)
#			plt.plot(pc.data[:,i])	
#		plt.show()
#		exit()


		# Iteration over all years of current DOY time series
		print('cnt = ',cnt,', neigs = ',neigs)
		if cnt == 366: nyears = 15
		for i in range(nyears):

			# Define current target day
			td_str = str(date_str[i])[:10]
			td = iData_Anomaly_Normalized.sel(time0=slice(td_str,td_str))

			# Transform TD field to iris cube
			iris_spfh_td = td.SPFH_GDS0_ISBL.to_iris()
			iris_prmsl_td = td.PRMSL_GDS0_MSL.to_iris()
			iris_rh_td = td.RH_GDS0_ISBL.to_iris()

			# Compute Pseudo-PCs
			ppc = msolver.projectField([iris_prmsl_td,iris_rh_td,iris_spfh_td],neofs=neigs)

			# Compute norm
			norm = []
			for j in range(pc.shape[0]):
				if td_str == str(date_arr[j][0])[:10]:
					#print(td_str,str(date_all_str[i])[:10])
					norm += [999999.]
				else:
					norm += [np.sqrt(np.sum((pc.data[j] - ppc.data[0])**2.0))]

			# Compute index of smallest norm
			min_index = np.argmin(norm) 
			norm_index_sort = np.argsort(norm)
			
			# redundant check
			if norm[norm_index_sort[0]] > norm[norm_index_sort[1]]:
				print("INVALID MINIMAL NORM!")
				exit()

			# Compute 5 best analogon
			analoga05 = []
			for ii in range(5):
				analoga05 += [str(date_arr[norm_index_sort[ii]][0]+datetime.timedelta(days=date_arr[norm_index_sort[ii]][1]-10))[:10]]


			newday_str = str(date_arr[min_index][0]+datetime.timedelta(days=date_arr[min_index][1]-10))[:10]
			print(i,':: ',td_str,':', newday_str)  

			# Compute RMSE values
			td_field = iData_Anomaly_Normalized.sel(time0=slice(td_str,td_str))
			nd_field = iData_Anomaly_Normalized.sel(time0=slice(newday_str,newday_str))

			rmse_spfh += [RMSE(td_field.SPFH_GDS0_ISBL.values[0,:,:],nd_field.SPFH_GDS0_ISBL.values[0,:,:])]
			rmse_prmsl += [RMSE(td_field.PRMSL_GDS0_MSL.values[0,:,:],nd_field.PRMSL_GDS0_MSL.values[0,:,:])]
			rmse_rh += [RMSE(td_field.RH_GDS0_ISBL.values[0,:,:],nd_field.RH_GDS0_ISBL.values[0,:,:])]
#			exit()

			# Write output
			file1.writerow([td_str,analoga05[0],analoga05[1],analoga05[2],norm[norm_index_sort[0]],norm[norm_index_sort[1]],norm[norm_index_sort[2]], rmse_spfh[-1],rmse_prmsl[-1],rmse_rh[-1]])




