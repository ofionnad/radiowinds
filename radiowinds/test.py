"""
This script is a quick test script that outputs a sample intensity plot and radio flux.
The parameters at the beginning are reasonable values to be chosen for an example grid of stellar wind.
"""
import radio_emission as re
import matplotlib.pyplot as plt

# set up some variables
ndim = 100          #size of arrays
gridsize = 10       #radius of grid boundary in rstar
n0 = 1e9            #base density
T0 = 2e6            #base temperature
gamma = 1.03        #polytropic index for test data
freq = 7e8          #observing frequency, set to 0.7 GHz
d = 10              #distance to star in pc
rstar = 1.0         #stellar radius in rsun units

# first calculate the integration constant, depends of stellar radius
int_c = re.integrationConstant(rstar)

#then we create a test dataset
# df is an array of [ds, n, T], where ds is out integration spacing (dx)
# and n and T are the 3d grid density and temeprature respectively.
dt = re.testData(ndim, gridsize, n0, T0, gamma, ordered=False)


#Then we remove the density from inside the star and behind the star
nt = re.emptyBack(dt[1], gridsize, ndim)


#Then we can calculate the intensity along the line of sight as well as flux density and radius of radio photosphere
I, sv, rv = re.radioEmission(dt[0], nt, dt[2], freq, d, ndim, gridsize, int_c)

#show the results
plt.show()

