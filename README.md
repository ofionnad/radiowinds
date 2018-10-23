# Radio Emission from Stellar Winds

This is a Python code to calculate the radio emission produced by the winds around stars. 
The code calculates thermal bremsstrahlung that is emitted from the wind, which depends directly on the density and temperature of the stellar wind plasma. 
The program takes input data in the form of an interpolated 3d grid of points (of the stellar wind) containing position, temperature and density data. 
From this it calculates the thermal free-free emission expected from the wind at a range of user-defined frequencies. 

This code is used in a paper currently submitted to Monthly Notices of the Royal Astronomical Society. 


## Installation
The code is available using pip:
`pip install radiowinds`

Or alternatively can be cloned directly from this repository.

## Testing
The quickest way to test that the code is working is to use the test script included in the package.

To test:
```python
from radiowinds import test

#set up initial parameters for grid
ndim = 50
gridsize = 10
ordered = True

#Instantiate testcase class
t = test.testcase(ndim, gridsize, ordered)
#call the test
data = t.test()

```
The `data` variable should now contain an array of 3 variables: 2d array of intensity, radio flux in Jy, and the size of the radio photosphere in R<sub>&#8902;</sub>.

The above test will also output an image that should look like the following:

![Alt text](radiowinds/test_ordered.png?raw=true "Thermal Bremstrahlung from a stellar wind")


## Quick Example Code
To use this code with your own data follow the steps below.
You require that the data is in the format of an evenly interpolated 3D grid.

There are many ways to interpolate a 3d grid of points (a function for which could be included at a later date here).
For the purposes of this example Tecplot was used to output an interpolated grid of points. 

The readData() function is used to get access to the data, it uses the pandas module. The radioEmission() function is the fastest way to make a calculation and get an output.
```python
import radiowinds.radio_emission as re

rstar = 1.05 #radius of star in R_sun

filename='/path/to/file.dat'
skiprows = 12 #this is the header size in the data file, which should be changed for users needs
ndim = 200 #This is the number of gridpoints in each dimension
gridsize = 10 #size of the radius of the grid in R_star

df = re.readData(filename, skiprows, ndim)

n = df[1] #grid density
T = df[2] #grid temperature
ds = df[0] #grid spacing along integration line
freq = 9e8 #observing frequency
dist = 10 #distance in parsecs

#remove density from behind star as this will not contribute to emisison
n = re.emptyBack(n, gridsize, ndim)

#find integration constant for grid of current size
int_c = re.integrationConstant(rstar)

I, sv, rv = re.radioEmission(ds, n, T, freq, dist, ndim, gridsize, int_c)
```
This should output an image of the intensity (and assign this data to `I`) from the wind and assign the radio flux to `sv` and the radio photopshere size to `rv`.

## Compute a Spectrum
This repository also provides a way to automatically cycle through a range of frequencies to find the spectrum of a stellar wind.

This can be done by using the `spectrumCalculate()` function.

Continuing on from the quick example above:

```python
#set a range of frequencies to iterate over
freqs = np.logspace(8,11,50)
output_dir = '/path/to/output' #where you want any output images to go
plotting=False #set to True to save images of intensity at each frequency to output_dir

svs, rvs = re.spectrumCalculate(output_dir, freqs, ds, n, T, d, ndim, gridsize, int_c, plotting=plotting)

```
`svs` will contain the flux in Jy at each frequency defined in freqs. To plot the spectrum simply use:
```python
plt.plot(freqs, svs)
```

### Creating animations
Using the images plotted from the spectrum function (provided `plotting == True`), one can use the moviepy module to make a short animation of the output.

```python
import radiowinds.make_animation as ma 

output_dir = '/path/to/output' #same directory as above

ma.make_animation(output_dir)
```
This will create an mp4 animation of the radio emission at different frequencies.

### Numerical Issues
Warning: 

If very low frequencies are used in the above calculations you run into some numerical problems.
Namely this is that the flux is overestimated.


### Author
Written by Dualta O Fionnagain in Trinity College Dublin, 2018
MIT License

Email: ofionnad@tcd.ie

Github: https://github.com/Dualta93
