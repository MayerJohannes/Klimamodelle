# 	Validation of the analog-method results using spartacus RR
#	David Hinger, Johannes Mayer
#	Klimamodelle SS19
#	version:	2019-07-23
# 
#	Execute:
#		python3 03-validation.py 1989-05-18 --plot


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
import scipy as sp
import pandas as pd
import numpy.ma as ma

# Parse target day as argument
parser = argparse.ArgumentParser(description='man')
parser.add_argument('td', type=str, help='[TD, e.g. 1989-05-18]', default=' ')
parser.add_argument('--plot', action='store_true', dest='splot', help='plot data')
args = parser.parse_args()
TD = args.td
sel_plt = args.splot

# Spartacus first and last year
syear = 1961
eyear = 2017


# Load date list (= EOF.py output)
with open('datelist-v4.csv', 'r') as f:
	reader = csv.reader(f)
	your_list = list(reader)

# Iterate over all entries until TD is found
for istr in your_list:
	if istr[0] == TD: break

str_td = istr[0] #'1981-01-01'
str_a1 = istr[1] #'2005-01-02'
str_a2 = istr[2] #'2012-01-04'

# Target day
year_td = str_td[:4]

# Analogon 1
year_a1 = str_a1[:4]

# Analogon 2
year_a2 = str_a2[:4]

ifile_td = '/home/johannes/klimamodelle/data/spartacus_rr/RR'+year_td+'.nc'
ifile_a1 = '/home/johannes/klimamodelle/data/spartacus_rr/RR'+year_a1+'.nc'
ifile_a2 = '/home/johannes/klimamodelle/data/spartacus_rr/RR'+year_a2+'.nc'

idata_td = xr.open_mfdataset(ifile_td,chunks={'time': 366})
idata_a1 = xr.open_mfdataset(ifile_a1,chunks={'time': 366})
idata_a2 = xr.open_mfdataset(ifile_a2,chunks={'time': 366})

td_field = idata_td.sel(time=slice(str_td,str_td))
a1_field = idata_a1.sel(time=slice(str_a1,str_a1))
a2_field = idata_a2.sel(time=slice(str_a2,str_a2))

td_values = td_field.RR.values[:,:][0]
a1_values = a1_field.RR.values[:,:][0]
a2_values = a2_field.RR.values[:,:][0]


# np.ravel : array to vector transformation
# masked_invalid : Mask an array where invalid values occur
# ma.corrcoef : compute corr coeff. matrix, considers missing values

# Correlation TD and best analogon
cc_a1 = ma.corrcoef(ma.masked_invalid(np.ravel(td_values)), ma.masked_invalid(np.ravel(a1_values)))[0,1]

# Correlation TD and second best analogon
cc_a2 = ma.corrcoef(ma.masked_invalid(np.ravel(td_values)), ma.masked_invalid(np.ravel(a2_values)))[0,1]
print(str_td, cc_a1, cc_a2)


# plot
plt.figure()
plt.subplot(3,1,1)
plt.title('TD: '+str_td)
plt.xticks([])
plt.yticks([])
cb = plt.pcolormesh(td_values,vmin=0,vmax=10)
cbar = plt.colorbar(cb)
cbar.set_label('mm/h')

plt.subplot(3,1,2)
plt.title('Analogon 1: '+str_a1+', ccoef = {:6.4f}'.format(cc_a1))
plt.xticks([])
plt.yticks([])
cb = plt.pcolormesh(a1_values,vmin=0,vmax=10)
cbar = plt.colorbar(cb)
cbar.set_label('mm/h')

plt.subplot(3,1,3)
plt.title('Analogon 2: '+str_a2+', ccoef = {:6.4f}'.format(cc_a2))
plt.xticks([])
plt.yticks([])
cb = plt.pcolormesh(a2_values,vmin=0,vmax=10)
cbar = plt.colorbar(cb)
cbar.set_label('mm/h')
if sel_plt: plt.show()




