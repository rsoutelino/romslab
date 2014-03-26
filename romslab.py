#!/usr/bin/env python
# Python Module to work with ROMS
# Owner: Rafael Soutelino - rsoutelino@gmail.com
# Committers: Andre Lobato - andrefelipelobato@gmail.com
# Last modification: Mar, 2012
#####################################################################
from dateutil.parser import parse
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import delaunay
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap
import scipy.io as sp  
import datetime as dt
import netCDF4 as nc

### CLASS RunSetup #################################################

def __version__():
    return "romslab-0.1"

class RunSetup(object):
    """Container class for ROMS runs metadata."""
    def __init__(self, filename):
        self.filename = filename
        f = open(filename)

        # skipping lines
        line = f.readline() 
        while line[0] == '*':
            line = f.readline() 

        # read and save header.
        self.header = ""
        while line[0] == '#':
            self.header += line[2:]
            key, value = line.split(':', 1) 
            key = key.strip()[2:]
            value = value.strip()
            # put all the information inside the class dictionary 
            self.__dict__[key.lower()] = value  

            line = f.readline() 
        
        # skipping lines.
        line = f.readline() 
        while line[0] == '*':
            line = f.readline() 
            
        # read and save grid information.
        while line[0] == '#':
            key, value = line.split(':', 2)
            key = key.strip()[2:]
            value = float(value)
            # put all the information inside the class dictionary
            self.__dict__[key.lower()] = value
            line = f.readline()
            
        # skipping lines.
        line = f.readline() 
        while line[0] == '*':
            line = f.readline()

        # read and save fields information.
        while line[0] == '#':
            key, value = line.split(':', 1) 
            key = key.strip()[2:]
            value = value.strip()       
            # put all the information inside the class dictionary 
            self.__dict__[key.lower()] = value  
            line = f.readline()
        
        # skipping lines.
        line = f.readline() 
        while line[0] == '*':
            line = f.readline()

        # read and save pathnames.
        while len(line) != 0 and line[0] == '#':
            key, value = line.split(':', 1) 
            key = key.strip()[2:]
            value = value.strip()       
            # put all the information inside the class dictionary 
            self.__dict__[key.lower()] = value  
            line = f.readline()

### CLASS RomsGrid ##################################################

class RomsGrid(object):
    """ 
    Stores and manipulates netcdf ROMS grid file information
    """
    def __init__(self,filename):
        self.filename = filename    
        self.ncfile = nc.Dataset(filename, mode='r+')
        self.lonr  = self.ncfile.variables['lon_rho'][:]
        self.latr  = self.ncfile.variables['lat_rho'][:]
        self.lonp  = self.ncfile.variables['lon_psi'][:]
        self.latp  = self.ncfile.variables['lat_psi'][:]
        self.lonu  = self.ncfile.variables['lon_u'][:]
        self.latu  = self.ncfile.variables['lat_u'][:]
        self.lonv  = self.ncfile.variables['lon_v'][:]
        self.latv  = self.ncfile.variables['lat_v'][:]
        self.maskr = self.ncfile.variables['mask_rho'][:]
        self.maskp = self.ncfile.variables['mask_psi'][:]
        self.masku = self.ncfile.variables['mask_u'][:]
        self.maskv = self.ncfile.variables['mask_v'][:]
    
    def corners(self):
        """
        Returns lon, lat cornes for a map projection:
        Usage: llclon, urclon, llclat, urclat = corners(self)
        """
        llclon = self.lonr.min()
        urclon = self.lonr.max()
        llclat = self.latr.min()
        urclat = self.latr.max()
        return llclon, urclon, llclat, urclat
        
### CLASS RomsHis ##################################################

class RomsHis(object):
    """ 
    Stores and manipulates netcdf ROMS history file information
    !!! Under construction !!!
    """
    def __init__(self,filename):
        self.filename = filename    
        self.ncfile = nc.Dataset(filename, mode='r+')
        self.lonr  = self.ncfile.variables['lon_rho'][:]
        self.latr  = self.ncfile.variables['lat_rho'][:]
        self.lonu  = self.ncfile.variables['lon_u'][:]
        self.latu  = self.ncfile.variables['lat_u'][:]
        self.lonv  = self.ncfile.variables['lon_v'][:]
        self.latv  = self.ncfile.variables['lat_v'][:]
        self.maskr = self.ncfile.variables['mask_rho'][:]
        self.masku = self.ncfile.variables['mask_u'][:]
        self.maskv = self.ncfile.variables['mask_v'][:]
    
    def corners(self):
        """
        Returns lon, lat cornes for a map projection:
        Usage: llclon, urclon, llclat, urclat = corners(self)
        """
        llclon = self.lonr.min()
        urclon = self.lonr.max()
        llclat = self.latr.min()
        urclat = self.latr.max()
        return llclon, urclon, llclat, urclat

### CLASS LoadEtopo5 ###################################################

class LoadEtopo5(object):
    """Reads and stores ETOPO 5 data"""
    def __init__(self):
        import netCDF4 as nc
        import numpy as np
        self.ncfile = nc.Dataset('/home/rsoutelino/misc/etopo5.nc')
        lon = self.ncfile.variables['topo_lon'][:]
        lat = self.ncfile.variables['topo_lat'][:]
        self.lon, self.lat = np.meshgrid(lon, lat)
        self.h = self.ncfile.variables['topo'][:]

### FUNCTION near ###################################################
				
def near(x,x0):
	"""
	Find the index where x has the closer value to x0
	"""
	
	dx = x - x0
	dx = abs(dx)
	fn = np.where( dx == dx.min() )
	
	return fn
	
### FUNCTION near2d ###################################################	

def near2d(x, y, x0, y0):
    """
    Find the indexes of the grid point that is
    nearest a chosen (x0, y0).
    Usage: line, col = near2d(x, y, x0, y0)
    """
    dx = np.abs(x - x0); dx = dx / dx.max()
    dy = np.abs(y - y0); dy = dy / dy.max()
    dn = dx + dy    
    fn = np.where(dn == dn.min())
    line = int(fn[0])
    col  = int(fn[1])
    return line, col
    
### FUNCTION subset ################################################### 

def subset(x, y, z, xmin, xmax, ymin, ymax):
    """
    Returns a subset z array based on x and y limits
    Usage: x2, y2, z2 = subset(x, y, z, xmin, xmax, ymin, ymax)
    """
    f = np.where((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) )
    x2 = x[ f[0][0]:f[0][-1], f[1][0]:f[1][-1] ]
    y2 = y[ f[0][0]:f[0][-1], f[1][0]:f[1][-1] ]
    z2 = z[ f[0][0]:f[0][-1], f[1][0]:f[1][-1] ]
    
    return x2, y2, z2

### FUNCTION ZLEV ###################################################

def zlev(h,theta_s,theta_b,Tcline,N,kgrid=0,zeta=0):
    """
    Set S-Curves in domain [-1 < sc < 0] 
    at vertical W- or RHO-points.

    On Input:  
	
    h         Bottom depth (m) of RHO-points (matrix).                     
    theta_s   S-coordinate surface control parameter (scalar):             
                [0 < theta_s < 20].                                        
    theta_b   S-coordinate bottom control parameter (scalar):              
                [0 < theta_b < 1].                                         
    Tcline    Width (m) of surface or bottom boundary layer in which       
              higher vertical resolution is required during streching      
              (scalar).                                                    
    N         Number of vertical levels (scalar).                          
    kgrid     Depth grid type logical switch:                              
                kgrid = 0   ->  depths of RHO-points.                      
                kgrid = 1   ->  depths of W-points.                        

    On Output:                                                                
	                                                                           
    z       Depths (m)    of RHO- or W-points (matrix).                    
    dz      Mesh size (m) at W-   or RHO-points (matrix).                  
    sc      S-coordinate independent variable, [-1 < sc < 0] at            
            vertical RHO-points (vector).                                  
    Cs      Set of S-curves used to stretch the vertical coordinate        
            lines that follow the topography at vertical RHO-points        
            (vector).  
    Copyright (c) 2003 UCLA - Patrick Marchesiello
    Translated to python by Rafael Soutelino - rsoutelino@gmail.com
    Last Modification: Aug, 2010
    """
    
    Np     = N + 1
    ds     = 1/N
    hmin   = h.min()
    hc     = min(hmin, Tcline)
    Mr, Lr = h.shape;

    if   kgrid==0:
	zeta = np.zeros([Mr, Lr])
	grid = 'r'
    elif zeta==0:
	zeta = np.zeros([Mr, Lr])
	
    if grid == 'r':
	Nlev = N
	lev  = np.arange(1, N+1, 1)
	sc   = -1 + (lev-0.5) * ds
    else:
	Nlev = Np
	lev  = np.arange(0, N+1, 1)
	sc   = -1 + lev * ds

	
    Ptheta = np.sinh(theta_s * sc) / np.sinh(theta_s)
    Rtheta = np.tanh(theta_s * (sc + 0.5) ) / (2*np.tanh(0.5 * theta_s)) -0.5
    Cs     = (1-theta_b)*Ptheta + theta_b*Rtheta
	
    cff0 = 1 + sc
    cff1 = (sc-Cs) * hc
    cff2 = Cs
    z  = np.zeros([N, Mr, Lr])
    for k in np.arange(0, Nlev, 1):
        z[k,:,:] = cff0[k]*zeta + cff1[k] + cff2[k]*h

    dz  = np.zeros([N, Mr, Lr])
    for k in np.arange(1, Nlev, 1):
        dz[k-1,:,:] = z[k,:,:] - z[k-1,:,:]
	
    return z, dz, sc, Cs
	



### FUNCTION ZTOSIGMA ###################################################

def ztosigma(var,z,depth):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Copyright (c) 2003 UCLA - Pierrick Penven                       %
    % Translated to Python by Rafael Soutelino, rsoutelino@gmail.com  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                 %
    % function  vnew = ztosigma(var,z,depth)                          %
    %                                                                 %
    % This function transform a variable from z to sigma coordinates  %
    %                                                                 %
    % On Input:                                                       %
    %                                                                 %
    %    var     Variable z (3D matrix).                              %
    %    z       Sigma depths (m) of RHO- or W-points (3D matrix).    %
    %    depth   z depth (vector; meters, negative).                  %
    %                                                                 %
    % On Output:                                                      %
    %                                                                 %
    %    vnew    Variable sigma (3D matrix).                          %
    %                                                                 %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Last Modification: Aug, 2010
    """

    Ns, Mp, Lp  = z.shape
    Nz          = depth.size
    vnew        = np.zeros([Ns, Mp, Lp])
    imat, jmat  = np.meshgrid(np.arange(1, Lp+1, 1), np.arange(1, Mp+1, 1))

    # Find the grid position of the nearest vertical levels

    for ks in np.arange(0, Ns, 1):
        sigmalev = np.squeeze(z[ks,:,:])
        thezlevs = 0 * sigmalev
	thezlevs = np.array(thezlevs, dtype=int)

        for kz in np.arange(0, Nz, 1):
	    f = np.where( sigmalev > depth[kz] )
	    thezlevs[f] = thezlevs[f] + 1

	pos = Nz * Mp * (imat-1) + Nz * (jmat-1) + thezlevs
	pos = np.array(pos, dtype=int)
	z1  = depth[thezlevs - 1]; z1 = np.squeeze(z1)
	z2  = depth[thezlevs];     z2 = np.squeeze(z2)
	var = np.ravel(var.transpose())
	pos = pos.ravel()
	v1  = var[pos-1]; v1.shape = (Mp,Lp)
	v2  = var[pos];   v2.shape = (Mp,Lp)
	vnew[ks,:,:] = (((v1-v2) * sigmalev + v2*z1 - v1*z2) / (z1-z2))

    return vnew



### FUNCTION RHO2UVP ###################################################

def rho2uvp(rfield):
    """
    ################################################################
    # 
    #   compute the values at u,v and psi points...
    # 
    #  Further Information:  
    #  http://www.brest.ird.fr/Roms_tools/
    #  
    #  This file is part of ROMSTOOLS
    #
    #  ROMSTOOLS is free software; you can redistribute it and/or modify
    #  it under the terms of the GNU General Public License as published
    #  by the Free Software Foundation; either version 2 of the License,
    #  or (at your option) any later version.
    #
    #  ROMSTOOLS is distributed in the hope that it will be useful, but
    #  WITHOUT ANY WARRANTY; without even the implied warranty of
    #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #  GNU General Public License for more details.
    #
    #  You should have received a copy of the GNU General Public License
    #  along with this program; if not, write to the Free Software
    #  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
    #  MA  02111-1307  USA
    #
    #  Copyright (c) 2001-2006 by Pierrick Penven 
    #  e-mail:Pierrick.Penven@ird.fr  
    # 
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
    #  Last Modification: Aug, 2010
    ################################################################
    """

    Mp, Lp  = rfield.shape
    M       = Mp - 1
    L       = Lp - 1
    
    vfield  = 0.5 * ( rfield[np.arange(0,M),:] + rfield[np.arange(1,Mp),:] )
    ufield  = 0.5 * ( rfield[:,np.arange(0,L)] + rfield[:,np.arange(1,Lp)] )
    pfield  = 0.5 * ( ufield[np.arange(0,M),:] + ufield[np.arange(1,Mp),:] )
    
    return ufield, vfield, pfield



### FUNCTION SPHERIC_DIST ###############################################

def spheric_dist(lat1,lat2,lon1,lon2):
    """
    #####################################################################
    #
    #  function dist=spheric_dist(lat1,lat2,lon1,lon2)
    #
    # compute distances for a simple spheric earth
    #
    #   input:
    #
    #  lat1 : latitude of first point (matrix)
    #  lon1 : longitude of first point (matrix)
    #  lat2 : latitude of second point (matrix)
    #  lon2 : longitude of second point (matrix)
    #
    #   output:
    #  dist : distance from first point to second point (matrix)
    # 
    #  Further Information:  
    #  http://www.brest.ird.fr/Roms_tools/
    #  
    #  This file is part of ROMSTOOLS
    #
    #  ROMSTOOLS is free software; you can redistribute it and/or modify
    #  it under the terms of the GNU General Public License as published
    #  by the Free Software Foundation; either version 2 of the License,
    #  or (at your option) any later version.
    #
    #  ROMSTOOLS is distributed in the hope that it will be useful, but
    #  WITHOUT ANY WARRANTY; without even the implied warranty of
    #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #  GNU General Public License for more details.
    #
    #  You should have received a copy of the GNU General Public License
    #  along with this program; if not, write to the Free Software
    #  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
    #  MA  02111-1307  USA
    #
    #  Copyright (c) 2001-2006 by Pierrick Penven 
    #  e-mail:Pierrick.Penven@ird.fr  
    #
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
    #  Last Modification: Aug, 2010
    ################################################################	
    """


    R = 6367442.76

#  Determine proper longitudinal shift.

    l = np.abs(lon2-lon1)
    l[np.where(l >= 180)] = 360 - l[np.where(l >= 180)]
                  
#  Convert Decimal degrees to radians.

    deg2rad = np.pi/180
    lat1    = lat1*deg2rad
    lat2    = lat2*deg2rad
    l       = l*deg2rad

#  Compute the distances

    dist   = R * np.arcsin( np.sqrt( ( (np.sin(l) * np.cos(lat2) )**2 )\
	+ (((np.sin(lat2) * np.cos(lat1)) - (np.sin(lat1) * np.cos(lat2)\
	* np.cos(l)))**2) ) )

    return dist


### FUNCTION GET_METRICS ###################################################

def get_metrics(latu, lonu, latv, lonv):
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #	Compute the pm and pn factors of a grid netcdf file 
    # 
    #  Further Information:  
    #  http://www.brest.ird.fr/Roms_tools/
    #  
    #  This file is part of ROMSTOOLS
    #
    #  ROMSTOOLS is free software; you can redistribute it and/or modify
    #  it under the terms of the GNU General Public License as published
    #  by the Free Software Foundation; either version 2 of the License,
    #  or (at your option) any later version.
    #
    #  ROMSTOOLS is distributed in the hope that it will be useful, but
    #  WITHOUT ANY WARRANTY; without even the implied warranty of
    #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #  GNU General Public License for more details.
    #
    #  You should have received a copy of the GNU General Public License
    #  along with this program; if not, write to the Free Software
    #  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
    #  MA  02111-1307  USA
    #
    #  Copyright (c) 2001-2006 by Pierrick Penven 
    #  e-mail:Pierrick.Penven@ird.fr  
    #
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
    #  Last Modification: Aug, 2010
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    Mp, L = latu.shape
    M, Lp = latv.shape
    Lm    = L - 1
    Mm    = M - 1

    dx    = np.zeros([Mp, Lp])
    dy    = np.zeros([Mp, Lp])
    dndx  = np.zeros([Mp, Lp])
    dmde  = np.zeros([Mp, Lp])

    lat1 = latu[:,np.arange(0,Lm)]
    lat2 = latu[:,np.arange(1,L)]
    lon1 = lonu[:,np.arange(0,Lm)]
    lon2 = lonu[:,np.arange(1,L)]

    dx[:,np.arange(1,L)] = spheric_dist(lat1, lat2, lon1, lon2)

    dx[:,0]    = dx[:,1]
    dx[:,Lp-1] = dx[:,L-1]

    lat1 = latv[np.arange(0,Mm),:]
    lat2 = latv[np.arange(1,M),:]
    lon1 = lonv[np.arange(0,Mm),:]
    lon2 = lonv[np.arange(1,M),:]

    dy[np.arange(1,M),:] = spheric_dist(lat1, lat2, lon1, lon2)

    dy[0,:]    = dy[1,:]
    dy[Mp-1,:] = dy[M-1,:]

    pm  = 1/dx
    pn  = 1/dy    

    #  dndx and dmde

    pn2 = pn[1:-2, 2:-1]; pn3 = pn[1:-2, 2:-1]
    dndx[1:-2, 1:-2] = 0.5 * (1/pn2 - 1/pn3)

    pm2 = pm[2:-1, 1:-2]; pm3 = pm[2:-1, 1:-2]
    dmde[1:-2, 1:-2] = 0.5 * (1/pm2 - 1/pm3)

    return pm, pn, dndx, dmde


### FUNCTION GET_ANGLE ###################################################
				
def get_angle(latu,lonu,argu1='wgs84'):
    """
    ################################################################
    #
    # Compute the grid orientation: angle [radians] 
    # between XI-axis and the direction to the EAST 
    # at RHO-points.
    #
    # lonu longitude of u points
    # latu latitude of u points
    # argu1: spheroid 
    #         'clarke66'  Clarke 1866
    #         'iau73'     IAU 1973
    #         'wgs84'     WGS 1984 (default)
    #         'sphere'    Sphere of radius 6371.0 km
    #
    # copied from dist.m of the Oceans toolbox
    #
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
    #  Last Modification: Aug, 2010
    #
    ################################################################
    """

    spheroid = argu1

    if spheroid[0:3] == 'sph':
	A   = 6371000.0
	B   = A
	E   = np.sqrt( A*A - B*B ) / A
	EPS = E*E / ( 1-E*E )
    elif spheroid[0:3] == 'cla': 
	A   = 6378206.4
	B   = 6356583.8
	E   = np.sqrt( A*A - B*B ) / A
	EPS = E*E / ( 1. - E*E )
    elif spheroid[0:3] == 'iau':
	A   = 6378160.0
	B   = 6356774.516
	E   = np.sqrt( A*A - B*B ) / A 
	EPS = E*E / ( 1. - E*E )
    elif spheroid[0:3] == 'wgs':
	A   = 6378137.0
	E   = 0.081819191
	B   = np.sqrt( A**2 - (A*E)**2)
	EPS = E*E / ( 1. - E*E )
    else:
	print  "Unknown spheroid was specified"

    latu = latu*np.pi / 180     # convert to radians
    lonu = lonu*np.pi / 180

    latu[np.where(latu == 0)] = 2.2204e-16  # Fixes some nasty 0/0 cases
    M, L = latu.shape

    PHI1  = latu[0:, 0:-1]    # endpoints of each segment
    XLAM1 = lonu[0:, 0:-1]
    PHI2  = latu[0:, 1:]
    XLAM2 = lonu[0:, 1:]

    # wiggle lines of constant lat to prevent numerical probs.
    f = np.where(PHI1 == PHI2)
    PHI2[f]  = PHI2[f]  + 1e-14

    # wiggle lines of constant lon to prevent numerical probs.
    f = np.where(XLAM1 == XLAM2)
    XLAM2[f] = XLAM2[f] + 1e-14

    # COMPUTE THE RADIUS OF CURVATURE IN THE PRIME VERTICAL FOR
    # EACH POINT

    xnu1 = A / np.sqrt( 1.0 - ( E * np.sin(PHI1) )**2 )
    xnu2 = A / np.sqrt( 1.0 - ( E * np.sin(PHI2) )**2 )

    # COMPUTE THE AZIMUTHS.  azim  IS THE AZIMUTH AT POINT 1
    # OF THE NORMAL SECTION CONTAINING THE POINT 2

    TPSI2 = ( 1.0 - E*E ) * np.tan(PHI2) + \
	E * E * xnu1 * np.sin(PHI1) / (xnu2 * np.cos(PHI2) )

    # SOME FORM OF ANGLE DIFFERENCE COMPUTED HERE??

    DLAM  = XLAM2 - XLAM1
    CTA12 = ( np.cos(PHI1) * TPSI2 - \
	np.sin(PHI1) * np.cos(DLAM) ) / np.sin(DLAM)
    azim  = np.arctan( 1.0 / CTA12 )

    #  GET THE QUADRANT RIGHT

    DLAM2 = ( np.abs(DLAM) < np.pi ) * DLAM + ( DLAM >= np.pi ) * \
	( (-2)*np.pi + DLAM ) + ( DLAM <= -np.pi ) * \
	( 2*np.pi + DLAM )

    azim  = azim + ( azim < -np.pi ) * 2*np.pi - \
	( azim >= np.pi ) * 2*np.pi

    azim  = azim + np.pi * np.sign(-azim) * \
	( np.sign(azim) != np.sign(DLAM2) )

    ang = np.zeros([M, L+1])

    ang[:,1:-1] = (np.pi/2) - azim
    ang[:,0]    = ang[:,1]
    ang[:,-1]   = ang[:,-2]

    return ang
		


### FUNCTION ADD_TOPO ###################################################

def add_topo(lon, lat, pm, pn, toponame):
    """
    ################################################################
    #
    # add a topography (here etopo2) to a ROMS grid
    #
    # the topogaphy matrix is coarsened prior
    # to the interpolation on the ROMS grid tp
    # prevent the generation of noise due to 
    # subsampling. this procedure ensure a better
    # general volume conservation.
    #
    # Last update Pierrick Penven 8/2006.
    #
    #    
    #  Further Information:  
    #  http://www.brest.ird.fr/Roms_tools/
    #  
    #  This file is part of ROMSTOOLS
    #
    #  ROMSTOOLS is free software; you can redistribute it and/or modify
    #  it under the terms of the GNU General Public License as published
    #  by the Free Software Foundation; either version 2 of the License,
    #  or (at your option) any later version.
    #
    #  ROMSTOOLS is distributed in the hope that it will be useful, but
    #  WITHOUT ANY WARRANTY; without even the implied warranty of
    #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #  GNU General Public License for more details.
    #
    #  You should have received a copy of the GNU General Public License
    #  along with this program; if not, write to the Free Software
    #  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
    #  MA  02111-1307  USA
    #
    #  Copyright (c) 2001-2006 by Pierrick Penven 
    #  e-mail:Pierrick.Penven@ird.fr 
    #
    #  Updated    Aug-2006 by Pierrick Penven
    #  Updated    2006/10/05 by Pierrick Penven (dl depend of model
    #                                           resolution at low resolution)
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
    #  Last Modification: Aug, 2010
    ################################################################
    """
    print '        Reading topog data'

    dat = sp.loadmat(toponame)
    x   = dat.pop('lon')
    y   = dat.pop('lat')
    z   = dat.pop('topo')

    dxt  = np.mean(np.abs(np.diff(np.ravel(x))))
    dyt  = np.mean(np.abs(np.diff(np.ravel(y))))
    dxt  = np.mean([dxt, dyt])

    if x.shape==y.shape==z.shape:
        pass
    else:
        x, y = np.meshgrid(x, y)

    print '        Slicing topog data into ROMS domain'

    # slicing topog into roms domain
    xm = np.ma.masked_where(x <= lon.min()-1, x)
    ym = np.ma.masked_where(x <= lon.min()-1, y)
    zm = np.ma.masked_where(x <= lon.min()-1, z)
    x  = np.ma.compress_cols(xm)
    y  = np.ma.compress_cols(ym)
    z  = np.ma.compress_cols(zm)

    del xm, ym, zm

    xm = np.ma.masked_where(x >= lon.max()+1, x)
    ym = np.ma.masked_where(x >= lon.max()+1, y)
    zm = np.ma.masked_where(x >= lon.max()+1, z)
    x  = np.ma.compress_cols(xm)
    y  = np.ma.compress_cols(ym)
    z  = np.ma.compress_cols(zm)

    del xm, ym, zm

    xm = np.ma.masked_where(y <= lat.min()-1, x)
    ym = np.ma.masked_where(y <= lat.min()-1, y)
    zm = np.ma.masked_where(y <= lat.min()-1, z)
    x  = np.ma.compress_rows(xm)
    y  = np.ma.compress_rows(ym)
    z  = np.ma.compress_rows(zm)

    del xm, ym, zm

    xm = np.ma.masked_where(y >= lat.max()+1, x)
    ym = np.ma.masked_where(y >= lat.max()+1, y)
    zm = np.ma.masked_where(y >= lat.max()+1, z)
    x  = np.ma.compress_rows(xm)
    y  = np.ma.compress_rows(ym)
    z  = np.ma.compress_rows(zm)

    del xm, ym, zm

    dxr = np.mean( 1/pm )
    dyr = np.mean( 1/pn )
    dxr = np.mean([dxr, dyr])
    dxr = np.floor(dxr/1852); dxr = dxr/60

    # degrading original topog resolution according to roms
    # grid resolution to avoid unecessary heavy computations 
     
    d  = int(np.floor( dxr/dxt )) 
    x  = x[0::d, 0::d]
    y  = y[0::d, 0::d]
    z  = z[0::d, 0::d]
    h = -z

    print '        Interp topog data into ROMS grid'

    h = griddata(x.ravel(),y.ravel(),h.ravel(),lon,lat,interp='nn')

    return h



### FUNCTION PROCESS_MASK ############################################

def process_mask(maskin):
	"""
	################################################################
	#
	#  maskout=process_mask(maskin)
	#
	#  Process the mask at rho-points in order to remove isolated
	#  masked points, cape with only 1 mask...
	#  Ensure continuous mask close to the boundaries
	# 
	#  Further Information:  
	#  http://www.brest.ird.fr/Roms_tools/
	#  
	#  This file is part of ROMSTOOLS
	#
	#  ROMSTOOLS is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation; either version 2 of the License,
	#  or (at your option) any later version.
	#
	#  ROMSTOOLS is distributed in the hope that it will be useful, but
	#  WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	#  MA  02111-1307  USA
	#
	#  Copyright (c) 2001-2006 by Pierrick Penven 
	#  e-mail:Pierrick.Penven@ird.fr  
	#  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
	#  Last Modification: Aug, 2010
	################################################################
	"""

	maskout = maskin.copy()

	M, L = maskout.shape
	Mm   = M - 1
	Lm   = L - 1
	Mmm  = Mm - 1
	Lmm  = Lm - 1

	neibmask = 0 * maskout
	neibmask[1:Mm, 1:Lm] = maskout[0:Mmm, 1:Lm] + maskout[2:M, 1:Lm] +\
						   maskout[1:Mm, 0:Lmm] + maskout[1:Mm, 2:L]

	neibint = neibmask[1:Mm, 1:Lm]
	maskint = maskout[1:Mm, 1:Lm]

	F1 = neibint * 0; F2 = maskint * 0; Fa = maskint * 0
	F1[ np.where(neibint >=3) ] = 1
	F2[ np.where(maskint ==0) ] = 1
	Fa = F1 == F2

	F1 = neibint * 0; F2 = maskint * 0; Fb = maskint * 0
	F1[ np.where(neibint <=1) ] = 1
	F2[ np.where(maskint ==1) ] = 1
	Fb = F1 == F2

	F = Fa.sum() + Fb.sum()

	c = 1

	if F > 0:
		f1 = neibmask >= 3
		f2 = maskout == 0 
		f  = f1 == f2
		maskout[f] = 1	 

		f1 = neibmask <= 1
		f2 = maskout == 1 
		f  = f1 == f2
		maskout[f] = 0	 


		maskout[0, 1:Lm] = maskout[1, 1:Lm]
		maskout[M-1, 1:Lm] = maskout[Mm-1, 1:Lm]
		maskout[1:Mm, 0] = maskout[1:Mm, 1]
		maskout[1:Mm, L-1] = maskout[1:Mm, Lm-1]

		maskout[0, 0]     = min( maskout[0, 1]     , maskout[1, 0]      )
		maskout[M-1, 0]   = min( maskout[M-1, 1]   , maskout[Mm-1, 0]   )
		maskout[0, L-1]   = min( maskout[0, Lm-1]  , maskout[1, L-1]    )
		maskout[M-1, L-1] = min( maskout[M-1, Lm-1], maskout[Mm-1, L-1] )



	# Be sure that there is no problem close to the boundaries

	maskout[:,0]   = maskout[:, 1]
	maskout[:,L-1] = maskout[:, Lm-1]
	maskout[0,:]   = maskout[1, :]
	maskout[M-1,:] = maskout[Mm-1, :]

	return maskout



### FUNCTION UVP_MASK ################################################

def uvp_mask(rfield):
	"""
	################################################################
	# 
	#   compute the mask at u,v and psi points... 
	# 
	#  Further Information:  
	#  http://www.brest.ird.fr/Roms_tools/
	#  
	#  This file is part of ROMSTOOLS
	#
	#  ROMSTOOLS is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation; either version 2 of the License,
	#  or (at your option) any later version.
	#
	#  ROMSTOOLS is distributed in the hope that it will be useful, but
	#  WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	#  MA  02111-1307  USA
	#
	#  Copyright (c) 2001-2006 by Pierrick Penven 
	#  e-mail:Pierrick.Penven@ird.fr  
	#
	#  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
	#  Last Modification: Aug, 2010
	################################################################
	"""

	Mp, Lp = rfield.shape
	M      = Mp-1
	L      = Lp-1

	vfield = rfield[0:M, :] * rfield[1:Mp, :]
	ufield = rfield[:, 0:L] * rfield[:, 1:Lp]
	pfield = ufield[0:M, :] * ufield[1:Mp, :]

	return ufield, vfield, pfield

### FUNCTION ZLEVS ###################################################

def zlevs(h,zeta,theta_s,theta_b,hc,N,type):
	"""
	################################################################
	#
	#  function z = zlevs(h,zeta,theta_s,theta_b,hc,N,type);
	#
	#  this function compute the depth of rho or w points for ROMS
	#
	#  On Input:
	#
	#    type    'r': rho point 'w': w point 
	#
	#  On Output:
	#
	#    z       Depths (m) of RHO- or W-points (3D matrix).
	# 
	#  Further Information:  
	#  http://www.brest.ird.fr/Roms_tools/
	#  
	#  This file is part of ROMSTOOLS
	#
	#  ROMSTOOLS is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation; either version 2 of the License,
	#  or (at your option) any later version.
	#
	#  ROMSTOOLS is distributed in the hope that it will be useful, but
	#  WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	#  MA  02111-1307  USA
	#
	#  Copyright (c) 2002-2006 by Pierrick Penven 
	#  e-mail:Pierrick.Penven@ird.fr  
	#  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
	#  Last Modification: Aug, 2010
	################################################################
	"""

	M, L = h.shape

	cff1 = 1.0 / np.sinh( theta_s )
	cff2 = 0.5 / np.tanh( 0.5*theta_s )

	if type=='w':
		sc = ( np.arange(0,N+1) - N ) / N
	  	N  = N + 1
	else:
		sc = ( np.arange(1,N+1) - N - 0.5 ) / N

	Cs = (1 - theta_b) * cff1 * np.sinh( theta_s * sc ) + \
		theta_b * ( cff2 * np.tanh(theta_s *(sc + 0.5) ) - 0.5 )

	hinv = 1 / h
	cff  = hc*( sc - Cs )
	cff1 = Cs
	cff2 = sc + 1
	z    = np.zeros([N, M, L])

	for k in range(0, N):
		z0         = cff[k] + cff1[k]*h
		z[k, :, :] = z0 + zeta*( 1 + z0*hinv )

	return z



### FUNCTION GET_DEPTHS ##############################################

def get_depths(fname,gname,tindex,type):
	"""
	######################################################################
	#
	#  Get the depths of the sigma levels
	#
	#  Further Information:  
	#  http://www.brest.ird.fr/Roms_tools/
	#  
	#  This file is part of ROMSTOOLS
	#
	#  ROMSTOOLS is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation; either version 2 of the License,
	#  or (at your option) any later version.
	#
	#  ROMSTOOLS is distributed in the hope that it will be useful, but
	#  WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	#  MA  02111-1307  USA
	#
	#  Copyright (c) 2002-2006 by Pierrick Penven 
	#  e-mail:Pierrick.Penven@ird.fr
	#  
	#  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
	#  Last Modification: Aug, 2010
	######################################################################
	"""
	
	def rho2u_3d(var_rho):
		N , Mp, Lp = var_rho.shape
		L = Lp - 1
		var_u = 0.5*( var_rho[:,:,0:L] + var_rho[:,:,1:Lp] )
		return var_u

	def rho2v_3d(var_rho):
		N, Mp, Lp = var_rho.shape
		M = Mp - 1
		var_v = 0.5*( var_rho[:,0:M,:] + var_rho[:,1:Mp,:])
		return var_v
		

	h    = gname.variables['h'][:]

	zeta    = np.squeeze( fname.variables['zeta'][tindex,...] )
	zeta[ np.where(zeta > 1e36) ] = 0
	
	theta_s = fname.variables['theta_s'][:]
	theta_b = fname.variables['theta_b'][:]
	Tcline  = fname.variables['Tcline'][:]
	hc      = fname.variables['hc'][:]
	s_rho   = fname.variables['s_rho'][:]
	hmin    = h.min()
	N       = s_rho.size

	vtype   = type

	if (type=='u') | (type=='v'):
		vtype='r'

	z  = zlevs(h,zeta,theta_s,theta_b,hc,N,vtype)

	if type=='u':
		z = rho2u_3d(z)

	if type=='v':
		z = rho2v_3d(z)

	return z


### FUNCTION SMOOTHGRID ##############################################

def smoothgrid(h,maskr,hmin,hmax_coast,rmax,n_filter_deep_topo,n_filter_final):
	"""
	#
	#  Smooth the topography to get a maximum r factor = rmax
	#
	#  n_filter_deep_topo:
	#  Number of pass of a selective filter to reduce the isolated
	#  seamounts on the deep ocean.
	#
	#  n_filter_final:
	#  Number of pass of a single hanning filter at the end of the
	#  procedure to ensure that there is no 2DX noise in the 
	#  topography.
	#
	#  Further Information:  
	#  http://www.brest.ird.fr/Roms_tools/
	#  
	#  This file is part of ROMSTOOLS
	#
	#  ROMSTOOLS is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation; either version 2 of the License,
	#  or (at your option) any later version.
	#
	#  ROMSTOOLS is distributed in the hope that it will be useful, but
	#  WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	#  MA  02111-1307  USA
	#
	#  Copyright (c) 2005-2006 by Pierrick Penven 
	#  e-mail:Pierrick.Penven@ird.fr  
	#
	#  Contributions of A. Shchepetkin (UCLA), P. Marchesiello (IRD)
	#                   and X. Capet (UCLA)
	#
	#  Updated    Aug-2006 by Pierrick Penven
	#
	#  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com 
	#  Last Modification: Aug, 2010
	################################################################
	"""
	# Cut the topography
	h[ np.where(h < hmin) ] = hmin

	# 1: Deep Ocean Filter
	if n_filter_deep_topo >= 1:
		print '        Removing isolated seamounts in the deep ocean'
		print '        ==> '+ str(n_filter_deep_topo) +' pass of selective filter'

	#	Build a smoothing coefficient that is a linear function 
	#	of a smooth topography.

		coef = h.copy()

		for i in range(1, 8):
			coef = hanning_smoother(coef)             # coef is a smoothed bathy
	   
		coef = 0.125 * (coef / coef.max() )           # rescale the smoothed bathy
	  
		for i in range(1, int(n_filter_deep_topo+1) ):
			h = hanning_smoother_coef2d(h, coef)      # smooth with a variable coef
			f1 = 0*maskr; f2 = 0*h 
			f1[ np.where(maskr == 0)]      = 1
			f2[ np.where(h > hmax_coast) ] = 1 
			f = f1 == f2
			h[f] = hmax_coast
	  
	
	print '        Applying filter on log(h) to reduce grad(h)/h'
	h = rotfilter(h, maskr, hmax_coast, rmax)

	#  Smooth the topography again to prevent 2D noise
	if n_filter_final > 1:
		
		print '        Smooth the topography a last time to prevent 2DX noise'
		print '        ==> '+ str(n_filter_final) +' pass of hanning smoother'
		
		for i in range( 1, int(n_filter_final) ):
			h = hanning_smoother(h)
			f1 = 0*maskr; f2 = 0*h 
			f1[ np.where(maskr == 0)]      = 1
			f2[ np.where(h > hmax_coast) ] = 1 
			f = f1 == f2
			h[f] = hmax_coast

	h[ np.where(h < hmin) ] = hmin

	return h


######################################################
def rotfilter(h, maskr, hmax_coast, rmax):
	"""
	#
	# Apply a selective filter on log(h) to reduce grad(h)/h.
	# 
	"""

	M, L   = h.shape
	Mm     = M - 1
	Mmm    = M - 2
	Lm     = L - 1
	Lmm    = L - 2
	cff    = 0.8
	nu     = 3.0/16.0
	rx, ry = rfact(h)
	r      = max(rx.max(), ry.max())
	h      = np.log(h)
	hmax_coast = np.log(hmax_coast)

	i = 0
	while r > rmax:
		i    = i + 1
		cx   = 0*rx; cy = 0*ry
		cx[ np.where(rx > cff*rmax) ] = 1
		cx   = hanning_smoother(cx)
		cy[ np.where(ry > cff*rmax) ] = 1
		cy   = hanning_smoother(cy)
		fx   = cx * FX(h)
		fy   = cy * FY(h)
		h[1:Mm, 1:Lm] = h[1:Mm, 1:Lm] + nu* \
						( (fx[1:Mm, 1:Lm] - fx[1:Mm, 0:Lmm] ) + \
						  (fy[1:Mm, 1:Lm] - fy[0:Mmm, 1:Lm] )   )
		h[0, :]   = h[1, :] 
		h[M-1, :] = h[Mm-1, :] 
		h[:, 0]   = h[:, 1] 
		h[:, L-1] = h[:, Lm-1]
		f1 = 0*maskr; f2 = 0*h 
		f1[ np.where(maskr == 0)]      = 1
		f2[ np.where(h > hmax_coast) ] = 1 
		f = f1 == f2
		h[f] = hmax_coast
	 
		rx, ry = rfact( np.exp(h) )
		r      = max(rx.max(), ry.max())
		print 'r factor = ' + str(r)


	h = np.exp(h)

	return h

##########################################################
def rfact(h):

	M, L = h.shape
	Mm   = M - 1
	Mmm  = M - 2
	Lm   = L - 1
	Lmm  = L - 2
	rx = np.abs( h[0:M, 1:L] - h[0:M, 0:Lm] ) / ( h[0:M, 1:L] + h[0:M, 0:Lm] )
	ry = np.abs( h[1:M, 0:L] - h[0:Mm, 0:L] ) / ( h[1:M, 0:L] + h[0:Mm, 0:L] )

	return rx, ry

##########################################################
def hanning_smoother(h):
	M, L = h.shape
	Mm   = M - 1
	Mmm  = M - 2
	Lm   = L - 1
	Lmm  = L - 2

	h[1:Mm, 1:Lm] = 0.125 * (  h[0:Mmm, 1:Lm] + h[2:M, 1:Lm] + \
                       h[1:Mm, 0:Lmm] + h[1:Mm, 2:L] + \
                       4 * h[1:Mm, 1:Lm]  )

	h[0, :]   = h[1, :]
	h[M-1, :] = h[Mm-1, :]
	h[:, 0]   = h[:, 1]
	h[:, L-1] = h[:, Lm-1]

	return h

##########################################################
def hanning_smoother_coef2d(h, coef):
	M, L = h.shape
	Mm   = M - 1
	Mmm  = M - 2
	Lm   = L - 1
	Lmm  = L - 2

	h[1:Mm, 1:Lm] = coef[1:Mm, 1:Lm] * ( h[0:Mmm, 1:Lm] + h[2:M, 1:Lm] + \
                                     h[1:Mm, 0:Lmm] + h[1:Mm,2:L]) + \
                            (1 - 4 * coef[1:Mm, 1:Lm]) * h[1:Mm, 1:Lm]
	h[0,:]   = h[1, :]
	h[M-1,:] = h[Mm-1, :]
	h[:,0]   = h[:, 1]
	h[:,L-1] = h[:, Lm-1]

	return h

##########################################################
def FX(h):
	M, L = h.shape
	Mm   = M - 1
	Mmm  = M - 2
	Lm   = L - 1
	Lmm  = L - 2
	
	fx = np.zeros([M, Lm])
	fx[1:Mm, :] = ( h[1:Mm, 1:L] - h[1:Mm, 0:Lm] ) * 5/6 + \
	   ( h[0:Mmm, 1:L] - h[0:Mmm, 0:Lm] + h[2:M, 1:L] - h[2:M, 0:Lm] ) / 12
   
	fx[0, :]   = fx[1, :]
	fx[M-1, :] = fx[Mm-1, :]

	return fx

##########################################################
def FY(h):
	M, L = h.shape
	Mm   = M - 1
	Mmm  = M - 2
	Lm   = L - 1
	Lmm  = L - 2

	fy = np.zeros([Mm, L])
	fy[:, 1:Lm] = ( h[1:M, 1:Lm] - h[0:Mm, 1:Lm] ) * 5/6 + \
        ( h[1:M, 0:Lmm] - h[0:Mm, 0:Lmm] + h[1:M, 2:L] - h[0:Mm, 2:L] ) / 12
	   
	fy[:, 0]   = fy[:, 1]
	fy[:, L-1] = fy[:, Lm-1]

	return fy

		
def wind_stress(u, v):
	"""
	function [taux,tauy]=wind_stress(u,v)                                
	                                                                     
	This function computes wind stress using Large and Pond formula.     
	                                                                     
	On Input:                                                            
	                                                                     
	   u         East-West wind component (m/s).                         
	   v         North-West wind component (m/s).                        
	                                                                     
	On Output:                                                           
	                                                                     
		taux      East-West wind stress component (Pa).                   
		tauy      East-West wind stress component (Pa).                   
	"""

	rhoa = 1.22
	speed = np.sqrt(u*u + v*v)
	Cd = (0.142 + 0.0764 * speed + 2.7 / (speed+0.000001)) * 0.001
	taux=rhoa * Cd * speed * u
	tauy=rhoa * Cd * speed * v

	return taux, tauy
                                                                
	
