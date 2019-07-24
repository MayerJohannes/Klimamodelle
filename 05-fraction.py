# 	Calculation of the fraction where the first corr.coeff is larger than the second one
#	David Hinger, Johannes Mayer
#	Klimamodelle SS19
#	version:	2019-07-24
# 
#	Execute:
#		python3 05-fraction.py

import csv

with open('corrcoef.csv','r') as f:
	reader = csv.reader(f)
	cclist = list(reader)

cctotal = 0
ana1 = 0

for cc in cclist:
	if cc[1] == '--' or cc[2] == '--': continue
	if cc[1] > cc[2]: ana1 = ana1 + 1
	cctotal = cctotal + 1

print(ana1/cctotal)

