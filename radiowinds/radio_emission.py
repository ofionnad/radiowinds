import numpy as np
import pandas as pd
import scipy.integrate as intg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import scipy.ndimage.interpolation as interpol
import scipy.spatial as sp
import decimal
import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *
import os

rsun = 6.957e10  # cm

def generateinterpolatedGrid(layfile, points, coords):
    """
    Function to create and save an interpolated tecplot simulation grid for radio emission calculation
    This will only work with Tecplot 360 installed on your system.

    :param layfile: Tecplot .lay file to be interpolated
    :param points: Number of points in each spatial dimension
    :param coords: Size of the grid in Rstar
    :return:
    """
    cwd = os.getcwd()
    tp.load_layout(layfile)
    frame1 = tp.active_frame()
    cur_dataset = frame1.dataset
    zone1 = cur_dataset.zone(0)  # zone1 is what tecplot uses for plotting in the layfile

    tp.macro.execute_command('''$!CreateRectangularZone
      IMax = {0:}
      JMax = {0:}
      KMax = {0:}
      X1 = -{1:}
      Y1 = -{1:}
      Z1 = -{1:}
      X2 = {1:}
      Y2 = {1:}
      Z2 = {1:}
      XVar = 1
      YVar = 2
      ZVar = 3'''.format(points, coords))

    zone2 = cur_dataset.zone(1)  # name second zone for interpolation
    tp.data.operate.interpolate_linear(zone2, source_zones=zone1, variables=[3, 10, 22])
    # create second zone and fill with variables
    tp.data.save_tecplot_ascii(cwd + '/interpol_grid_{0:}Rstar_{1:}points.dat'.format(coords, points),
                               zones=[zone2],
                               variables=[0, 1, 2, 3, 10, 22],
                               include_text=False,
                               precision=9,
                               include_geom=False,
                               use_point_format=True)
    return

def integrationConstant(rstar):
    """
    Function to set the integration constant, based off the stellar radius.

    :param rstar: the radius of the star in units of rsun
    :return: integration constant, int_c
    """
    int_c = rstar * rsun
    return int_c

def testData(points, gridsize, n0, T0, gam, ordered=True):
    """
    Function to produce a grid of sample values of density and temperature.
    Either ordered which follows a n ~ R^{-3} profile, or not ordered which has a more randomised distribution.

    :param points: Number of gridpoints in each dimension
    :param gridsize: The size of the grid radius in rstar
    :param n0: base density of the stellar wind
    :param T0: base temperature of the stellar wind
    :param gam: polytopic index of the wind to derive temperature from density
    :param ordered: either cause density to fall off with R^{-3} or be more randomised with a R^{-3} component
    :return: ds, n, T. ds is the spacing in the grid used for integration. n is the grid density (shape points^3). T is the grid temperature (shape points^3).
    """
    if ordered==True:
        o = np.array([int(points / 2), int(points / 2), int(points / 2)])
        x = np.linspace(0, points - 1, points)
        y = np.linspace(0, points - 1, points)
        z = np.linspace(0, points - 1, points)
        X, Y, Z = np.meshgrid(x, y, z)
        ds = np.linspace(-gridsize,gridsize,points)
        d = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        sph_dist = sp.distance.cdist(d, o.reshape(1, -1)).ravel()
        sph_dist = sph_dist.reshape(points, points, points) /(points/2/gridsize)
        sph_dist[int(points/2), int(points/2), int(points/2)] = 1e-40
        n = n0 * (sph_dist ** -3)
        n[int(points / 2), int(points / 2), int(points / 2)] = 0
        # this is getting rid of the centre inf, doesn't matter as it is at centre and is removed anyway!
        T = T0 * (n / n0) ** gam
        return ds, n, T
    else:
        o = np.array([int(points / 2), int(points / 2), int(points / 2)])
        x = np.linspace(0, points - 1, points)
        y = np.linspace(0, points - 1, points)
        z = np.linspace(0, points - 1, points)
        X, Y, Z = np.meshgrid(x, y, z)
        ds = np.linspace(-gridsize, gridsize, points)
        d = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        sph_dist = sp.distance.cdist(d, o.reshape(1, -1)).ravel()
        sph_dist = sph_dist.reshape(points, points, points) / (points / 2 / gridsize)
        sph_dist[int(points/2), int(points/2), int(points/2)] = 1e-20

        #make random array of data
        rand_n = n0*np.random.rand(points,points,points)
        n = rand_n * (sph_dist ** -3) #give it a resemblence of falling off with distance
        n[n>1e8] = n[n>1e8]+2e8  #cause some change to increase centre contrast in density!
        n[int(points / 2), int(points / 2), int(points / 2)] = 0
        # this is getting rid of the centre inf, doesn't matter as it is at centre and is removed anyway!

        T = T0 * (n / n0) ** gam
        return ds, n, T

def readData(filename, skiprows, points):
    """
    This function expects an interpolated grid of data. Originally interpolated using the tecplot software.
    Not tested yet but I am sure VisIT interpolated produced a similar output and can also be used.
    Maybe include grid interpolation function in future.

    :param filename: Name of the data file to read from
    :param skiprows: Number of rows to skip (according to the pandas read_csv() function.
    :param points: Number of gridpoints in each dimension
    :return: ds, n, T. Grid spacing, grid density and grid temperature.
    """
    df = pd.read_csv(filename, header=None, skiprows=skiprows, sep='\s+')
    X = df[0].values.reshape((points, points, points))
    ds = X[0, 0, :]
    n_grid = (df[3].values.reshape((points, points, points))) / (1.673e-24 * 0.5)
    T_grid = df[5].values.reshape((points, points, points))
    return ds, n_grid, T_grid

def rotateGrid(n, T, degrees, axis='z'):
    """
    Function that rotates the grid so that the emission can be calculated from any angle.

    :param n: grid densities
    :param T: grid temperatures
    :param degrees: number of degrees for grid to rotate. Can be negative or positive, will rotate opposite directions
    :param axis: This keyword sets the axis to rotate around. Default is z. A z axis rotation will rotate the grid "left/right". An x-axis rotation would rotate the grid "forwards/backwards" and should be used to set inclination of star.
    :return: n and T, rotated!
    """

    # The z axis rotates the grid around the vertical axis (used for rotation modulation of a star for example)
    if axis == 'z':
        n_rot = interpol.rotate(n, degrees, axes=(1, 2), reshape=False)
        T_rot = interpol.rotate(T, degrees, axes=(1, 2), reshape=False)
        return n_rot, T_rot

    # The x axis rotates the grid around the horizontal axis (used for tilting for stellar inclinations)
    if axis == 'x':
        n_rot = interpol.rotate(n, degrees, axes=(0, 2), reshape=False)
        T_rot = interpol.rotate(T, degrees, axes=(0, 2), reshape=False)
        return n_rot, T_rot

    # The following is only included for completeness, you should never need to rotate around this axis!!!
    if axis == 'y':
        n_rot = interpol.rotate(n, degrees, axes=(0, 1), reshape=False)
        T_rot = interpol.rotate(T, degrees, axes=(0, 1), reshape=False)
        return n_rot, T_rot

    else:
        print("axis is type: ", type(axis))
        raise ValueError("Axis is the wrong type")
        pass

def emptyBack(n, gridsize, points):
    """
    Function that sets the density within and behind the star to zero (or very close to zero).

    :param n: grid densities
    :param gridsize: size of grid radius in rstar
    :param points: number of gridpoints in each dimension
    :return: n, the original grid of densities with the necessary densities removed
    """
    # First block of code removes the densities from the sphere in the centre
    points = int(points)
    c = points / 2  # origin of star in grid
    o = np.array([c, c, c])  # turn into 3d  vector origin
    rad = points / (gridsize * 2)  # radius of star in indices
    x1 = np.linspace(0, points - 1, points)  # indices array, identical to y1 and z1
    y1 = x1
    z1 = x1
    X, Y, Z = np.meshgrid(x1, y1, z1)  # make 3d meshgrid
    d = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # a 2d array of all of all the coordinates in the grid
    sph_dist = sp.distance.cdist(d, o.reshape(1, -1)).ravel()  # the distance of each coordinate from the origin
    p_sphere = d[sph_dist < rad]  # the indices that exist inside the star at the centre
    p_sphere = p_sphere.astype(int, copy=False)  # change index values to integers
    for i in p_sphere:
        n[i[0], i[1], i[2]] = 1e-40

    # Now remove the cyclinder behind the star which is out of sight.
    o2 = o[:2]  # the 2d centre of the xz plane
    d2 = np.vstack((X.ravel(), Y.ravel())).T  # converting the indices into a 1d array of points
    circ_dist = sp.distance.cdist(d2,
                                  o2.reshape(1, -1)).ravel()  # find the distance of all points in d2 from the origin o2
    p_circ = d2[circ_dist < rad]  # find the indices of points inside the circle
    p_circ = p_circ.astype(int, copy=False)
    for i in range(int(
            points / 2)):  # iterate over the xz planes moving back from centre of the grid (points/2) to the back (points)
        for j in p_circ:
            n[int(j[0]), int(i+c), int(j[1])] = 1e-40
            #n[int(j[0]), int(j[1]), int(i + c)] = 1e-40

    return n

def absorptionBody(n, T, f):
    """
    Function that calculates the absorption coefficients and blackbody emission value
    for each cell in the interpolated tecplot grid

    :param n: density of cell
    :param T: temperature of cell
    :param f: observing frequency

    :return: alpha_v, B(v,T) : absorption coefficients and the blackbody of each cell
    """
    gaunt = 10.6 + (1.90 * np.log10(T)) - (1.26 * np.log10(f))
    kb = 1.38e-16
    h = 6.62607e-27
    c = 2.998e10
    absorption_c = 3.692e8 * (1.0 - np.exp(-(h * f) / (kb * T))) * ((n**2 / 4.)) * (T ** -0.5) * (f ** -3.0) * gaunt
    bb = ((2.0 * h * (f ** 3.0)) / (c ** 2.0)) * (1.0 / (np.exp((h * f) / (kb * T)) - 1.0))
    absorption_c[np.isnan(absorption_c)] = 1e-40
    absorption_c[np.isinf(absorption_c)] = 1e-40
    return absorption_c, bb

def get_gaunt(T, f):
    """
    Function that simply returns grid of values of gaunt factors from temperatures and frequencies
    Note: Assumes that Z (ionic charge) is +1.

    :param T: grid of temperatures in Kelvin
    :param f: observational frequency
    :return: grid of gaunt factors the same shape as T
    """

    gaunt = 10.6 + (1.90 * np.log10(T)) - (1.26 * np.log10(f))
    return gaunt

def opticalDepth(ds, ab, int_c):
    """
    Calculates the optical depth of material given the integration grid and the absorption coefficients.

    :param ds: The regular spacing of the interpolated grid (integration distances , ds)
    :param ab: grid of absorption coefficients calculated from absorptionBody()
    :param int_c: integration constant calculated from integrationConstant()

    :return: array of cumulative optical depth (tau)
    """
    tau = (intg.cumtrapz(ab, x=ds, initial=0, axis=1)) * int_c
    #note that axis=1 is here to ensure integration occurs along y axis.
    #Python denotes the arrays as [z,y,x] in Tecplot axes terms.(x and z axes notation are swapped)

    return tau

def intensity(ab, bb, tau, ds, int_c):
    """
        Name : intensity()

        Function : Calculates the intensity of emission given the blackbody emission from each grid cell and the optical depth at each cell
                   Note : Not sure whether to take the last 2d grid of cells (i.e. - Iv[:,:,-1])
                          or sum up each column given the bb and tau (i.e. - np.sum(Iv, axis=2).
    """
    I = intg.simps((bb * np.exp(-tau)) * ab, x=ds, axis=1) * int_c
    return I

def flux_density(I, ds, d, int_c):
    """
        Name : flux_density()

        Function : Calculates the flux density given a certain intensity and distance (pc)
    """
    d *= 3.085678e18  # change d from pc to cm
    # flux here given in Jy
    Sv = 1e23 * (int_c ** 2.0) * (intg.simps(intg.simps(I, x=ds), x=ds)) / d ** 2.0
    return Sv

def get_Rv(contour, points, gridsize):
    """
     Function to get the coordinates of a contour.
     Input:

        contour  : The contour object plotted on image
        points     : number of grid points in image
        gridsize : Size of grid in Rstar

     Returns:

        Rv - Size of radius of emission in Rstar

    """
    path = contour.collections[0].get_paths()
    path = path[0]
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]
    x1, y1 = x - points / 2, y - points / 2
    r = np.sqrt(x1 ** 2 + y1 ** 2)
    Rv = gridsize * (max(r) / (points / 2.0))
    return Rv

def double_plot(I, tau, f_i, points, gridsize):
    """
    Plot two images beside each other
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    p = ax1.imshow(I, interpolation='bilinear', origin='lower', norm=LogNorm(vmin=1e-20, vmax=1e-12), cmap=cm.Greens)
    fig.suptitle(r'$\nu_{{\rm ob}}$ =  {0:.2f} Hz'.format(f_i), bbox=dict(fc="w", ec="C3", boxstyle="round"))
    circ1 = plt.Circle((points / 2, points / 2), (points / (2 * gridsize)), color='white', fill=True, alpha=0.4)
    ax1.add_artist(circ1)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="8%", pad=0.1)
    cbar1 = plt.colorbar(p, cax=cax1)
    cbar1.set_label(r'I$_{\nu}$ (erg/s/cm$^2$/sr/Hz)', fontsize=16)
    p2 = ax2.imshow(tau[:, -1, :], interpolation='bilinear', origin='lower', norm=LogNorm(vmin=1e-8, vmax=0.399), cmap=cm.Oranges)
    circ2 = plt.Circle(((points) / 2, (points) / 2), (points / (2 * gridsize)), color='white', fill=True, alpha=0.4)
    ax2.add_artist(circ2)
    cset1 = ax2.contour(tau[:, -1, :], [0.399], colors='k', origin='lower', linestyles='dashed')
    Rv_PF = (get_Rv(cset1, points, gridsize))
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="8%", pad=0.1)
    cbar2 = plt.colorbar(p2, cax=cax2)
    cbar2.set_label(r'$\tau_{\nu}$', fontsize=16)
    plt.tight_layout()
    ax1.set_xticks(np.linspace(0, 200, 5))
    ax2.set_xticks(np.linspace(0, 200, 5))
    ax1.set_yticks(np.linspace(0, 200, 5))
    ax2.set_yticks(np.linspace(0, 200, 5))
    ax1.set_xticklabels(['-10', '-5', '0', '5', '10'], fontsize=12)
    ax1.set_yticklabels(['-10', '-5', '0', '5', '10'], fontsize=12)
    ax2.set_xticklabels(['-10', '-5', '0', '5', '10'], fontsize=12)
    ax2.set_yticklabels(['-10', '-5', '0', '5', '10'], fontsize=12)
    ax1.set_xlim([25, 175])
    ax2.set_xlim([25, 175])
    ax1.set_ylim([25, 175])
    ax2.set_ylim([25, 175])
    ax1.set_ylabel(r'R$_{\star}$', fontsize=16)
    ax1.set_xlabel(r'R$_{\star}$', fontsize=16)
    ax2.set_ylabel(r'R$_{\star}$', fontsize=16)
    ax2.set_xlabel(r'R$_{\star}$', fontsize=16)
    ax1.grid(which='major', linestyle=':', alpha=0.8)
    ax2.grid(which='major', linestyle=':', alpha=0.8)
    plt.show()
    #plt.close()
    return Rv_PF

def spectrumCalculate(folder, freqs, ds, n_i, T_i, d, points, gridsize, int_c, plotting=False):
    """
    Inputs : folder name, range of frequencies, position coordinate, density, temperature

    Function : Calculates flux density (Sv) and radius of emission (Rv) for a range of frequencies
    """
    print(d)
    Svs = []
    Rvs = []
    taus = []
    for i, j in enumerate(freqs):
        ab, bb = absorptionBody(n_i, T_i, j)
        tau = opticalDepth(ds, ab, int_c)
        taus.append(np.mean(tau))
        I = intensity(ab, bb, tau, ds, int_c)
        Sv = flux_density(I, ds, d, int_c)
        Svs.append(Sv)
        Rv = double_plot(I, tau, j, points, gridsize)
        Rvs.append(Rv)
        if not plotting:
            pass
        else:
            Rv, ax = single_plot(I, tau, j, points, gridsize)
            plt.savefig('{0:}/img_{1:}'.format(folder, i), dpi=500)
            plt.close('all')
    return Svs, Rvs

def radioEmission(ds, n_i, T_i, f, d, points, gridsize, int_c):
    """
    Inputs : position coordinates, density (/cc), Temperature (K), distance to star (pc), number of points in each axes of the grid, grid size in rstar.

    Output : Flux density (Sv), Radius of emission (Rv), plots intensity and optical depth.
    """
    ab, bb = absorptionBody(n_i, T_i, f)
    tau = opticalDepth(ds, ab, int_c)
    print(tau[:,-1,:].max(), tau[:,-1,:].min())
    I = intensity(ab, bb, tau, ds, int_c)
    Sv = flux_density(I, ds, d, int_c)
    Rv, ax = single_plot(I, tau, f, points, gridsize)
    return I, Sv, Rv

def single_plot(I, tau, f, points, gridsize):
    """
    Plot a single intensity image with the contours from the relevant optical depth image
    :param I: Intensities
    :param tau: 2d array of optical depths
    :param f: observing frequency
    :param points: number of points in the grid in each spatial dimension
    :param gridsize: the size of the grid in rstar

    :return: Rv_PF - the radius of the optically thick regionax1 - the axes of the plot that is shown

    """

    fig, axs = plt.subplots(1, 1, figsize=(7.3, 6))
    p2 = axs.imshow(I, interpolation='bilinear', origin='lower', norm=LogNorm(vmin=1e-17, vmax=1e-12), cmap=cm.Greens)
    cset1 = plt.contour(tau[:, -1, :], []0.399], colors='k', origin='lower', linestyles='dashed')
    frequency_text = int(f)
    plt.text(int(points/10), int(points-(points/10)), r'$\nu_{{\rm ob}}$ =  {}'.format(prettyprint(frequency_text, 'Hz')),
             bbox=dict(fc="w", ec="C3", boxstyle="round", alpha=0.8), fontsize=12)
    circ3 = plt.Circle(((points) / 2, (points) / 2), (points / (2 * gridsize)), color='white', fill=True, alpha=0.4)
    axs.add_artist(circ3)
    divs = make_axes_locatable(axs)
    caxs = divs.append_axes("right", size="8%", pad=0.1)
    cbars = plt.colorbar(p2, cax=caxs)
    cbars.set_label(r'I$_{\nu}$ (erg s$\rm ^{-1} cm^{-2} sr^{-1} Hz^{-1}$)', fontsize=20)
    cbars.ax.tick_params(labelsize=15)
    if tau[:,-1,:,].min() > 0.399:
        print("\nNo contours! - Optically Thick Wind - Numerical issues in this regime!")
        Rv_PF = None
    elif tau[:,-1,:].max() < 0.399:
        print("\nNo contours! Optically Thin Wind")
        Rv_PF = None
    else:
        Rv_PF = get_Rv(cset1, points, gridsize)

    axs.set_xticks(np.linspace(0, points, 5))
    axs.set_yticks(np.linspace(0, points, 5))
    axs.set_xticklabels(['-'+str(int(gridsize)), '-'+str(int(gridsize/2)), '0', str(int(gridsize/2)), str(int(gridsize))], fontsize=16)
    axs.set_yticklabels(['-'+str(int(gridsize)), '-'+str(int(gridsize/2)), '0', str(int(gridsize/2)), str(int(gridsize))], fontsize=16)
    #axs.set_xlim([int((points / 2) - (7.5 * (points / (2 * gridsize)))), int((points / 2) - (15 * (points / (2 * gridsize))))])
    #axs.set_ylim([int((points / 2) - (7.5 * (points / (2 * gridsize)))), int((points / 2) - (15 * (points / (2 * gridsize))))])
    axs.set_xlabel(r'r (R$_{\star}$)', fontsize=20)
    axs.set_ylabel(r'r (R$_{\star}$)', fontsize=20)
    axs.grid(which='major', linestyle=':', alpha=0.8, color='white')
    plt.tight_layout()

    return Rv_PF, axs

def prettyprint(x, baseunit):
    """
    Just a function to round the printed units to nice amounts

    :param x: Input value
    :param baseunit: Units used
    :return: rounded value with correct unit prefix
    """
    prefix = 'yzafpnµm kMGTPEZY'
    shift = decimal.Decimal('1E24')
    d = (decimal.Decimal(str(x)) * shift).normalize()
    m, e = d.to_eng_string().split('E')
    m = "{0:.2f}".format(float(m))
    return m + " " + prefix[int(e) // 3] + baseunit
