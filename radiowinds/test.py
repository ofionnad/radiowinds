"""
This script is a quick test script that outputs a sample intensity plot and radio flux.
The parameters at the beginning are reasonable values to be chosen for an example grid of stellar wind.
"""
import radiowinds.radio_emission as re
import matplotlib.pyplot as plt


class testcase():

    def __init__(self, ndim, gridsize, ordered):
        """
        Initialize required parameters for the test case.

        :param ndim: number of points in each dimension in the grid
        :param gridsize: size of the radius of the grid in rstar
        :param ordered: True for rho ~ R^-3 or False for more randomised grid.
        """
        self.ndim = ndim
        self.gridsize = gridsize
        self.n0 = 1e9
        self.T0 = 2e6
        self.gamma = 1.05
        self.freq = 7e8
        self.d = 10
        self.rstar = 1.0
        self.ordered = ordered

    def test(self):
        """
        Runs a test case to see if the code can make a grid and calculate thermal radio emission from it.
        Really just a test to see if all packaged are installed, probably not the best way to do this.
        Can be compared with output example on GitHub.

        :return: array of 2d intensity, radio flux in Jy, size of radio photosphere in rstar.
        """
        # first calculate the integration constant, depends of stellar radius
        int_c = re.integrationConstant(self.rstar)
        dt = re.testData(self.ndim, self.gridsize, self.n0, self.T0, self.gamma, ordered=self.ordered)
        nt = re.emptyBack(dt[1], self.gridsize, self.ndim)
        I, sv, rv = re.radioEmission(dt[0], nt, dt[2], self.freq, self.d, self.ndim, self.gridsize, int_c)
        plt.show()
        return I, sv, rv

"""    
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

"""