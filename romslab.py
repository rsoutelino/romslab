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
import scipy.interpolate as spint
from mpl_toolkits.basemap import Basemap
import scipy.io as sp  
import datetime as dt
import netCDF4 as nc
import yaml

### CLASS RunSetup #################################################

def __version__():
    return "romslab-0.2"

class RunSetup(object):
    """ROMS domain config file"""
    def __init__(self, filename, imp):
        self.filename = filename
        self.id = imp
        f =  yaml.load( open(filename) )
        configs = f[imp]
        for key in configs.keys():
            execstr = "self.%s = configs[key]" %key
            exec(execstr)
        
        self.lonmin, self.lonmax = self.lims[0], self.lims[1]
        self.latmin, self.latmax = self.lims[2], self.lims[3]
        self.hmaxc = self.hmin

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
        self.lonu  = self.ncfile.variables['lon_u'][:]
        self.latu  = self.ncfile.variables['lat_u'][:]
        self.lonv  = self.ncfile.variables['lon_v'][:]
        self.latv  = self.ncfile.variables['lat_v'][:]
        self.h     = self.ncfile.variables['h'][:]
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
        
### CLASS RomsHis ##############################################

class RomsHis(object):
    """ 
    Stores and manipulates netcdf ROMS history file information
    !!! Under construction !!!
    """
    def __init__(self,filename):
        self.filename = filename    
        self.ncfile = nc.Dataset(filename, mode='r')
        self.varlist = list(self.ncfile.variables)
        
        for var in self.varlist:
            exec("self.%s = self.ncfile.variables['%s']" %(var, var) )
#        self.lonr  = self.ncfile.variables['lon_rho'][:]
#        self.latr  = self.ncfile.variables['lat_rho'][:]
#        self.lonu  = self.ncfile.variables['lon_u'][:]
#        self.latu  = self.ncfile.variables['lat_u'][:]
#        self.lonv  = self.ncfile.variables['lon_v'][:]
#        self.latv  = self.ncfile.variables['lat_v'][:]
#        self.maskr = self.ncfile.variables['mask_rho'][:]
#        self.masku = self.ncfile.variables['mask_u'][:]
#        self.maskv = self.ncfile.variables['mask_v'][:]
    
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

### CLASS M2_diagnostics #####################################################

class M2_diagnostics(object):
    """
    Container class for the depth-averaged (2D) momentum equation
    diagostic terms for a ROMS run.

    USAGE
    -----
    m2_terms = M2_diagnostics(diafile, verbose=False)

    diafile is a ROMS diagnostics file (*_dia.nc). This class extracts the
    M2 diagnostic terms provided that all are present in the file.

    Returns an object with a '.xi' and a '.eta' attribute. Those are dictionaries
    that store the <netCDF4.Variable> objects corresponding to each term.
    """
    def __init__(self, diafile):
        dia = nc.Dataset(diafile)
        self.xi = dict()
        self.eta = dict()
        self.xi_labels = dict()
        self.eta_labels = dict()
        ## Terms of the M2 balance in the XI-component.
        self.xi['ut'] = dia.variables['ubar_accel']
        self.xi['uux'] = dia.variables['ubar_xadv']
        self.xi['vuy'] = dia.variables['ubar_yadv']
        self.xi['ucor'] = dia.variables['ubar_cor']
        self.xi['upgrd'] = dia.variables['ubar_prsgrd']
        self.xi['uistr'] = dia.variables['ubar_hvisc']
        self.xi['usstr'] = dia.variables['ubar_sstr']
        self.xi['ubstr'] = dia.variables['ubar_bstr']
        ## Terms of the M2 balance in the ETA-component.
        self.eta['vt'] = dia.variables['vbar_accel']
        self.eta['uvx'] = dia.variables['vbar_xadv']
        self.eta['vvy'] = dia.variables['vbar_yadv']
        self.eta['vcor'] = dia.variables['vbar_cor']
        self.eta['vpgrd'] = dia.variables['vbar_prsgrd']
        self.eta['vistr'] = dia.variables['vbar_hvisc']
        self.eta['vsstr'] = dia.variables['vbar_sstr']
        self.eta['vbstr'] = dia.variables['vbar_bstr']
        self.diafile = dia
        self.RUN_AVERAGED = False

        ## Move all fields to PSI-points.
        print "Moving all fields to PSI-points."
        for term in self.xi.iterkeys():
            self.xi[term] = 0.5*(self.xi[term][:,1:,:]+self.xi[term][:,:-1,:])
        for term in self.eta.iterkeys():
            self.eta[term] = 0.5*(self.eta[term][:,:,1:]+self.eta[term][:,:,:-1])

        self.nt = self.xi['ut'].shape[0]
        self.x = self.diafile.variables['lon_psi'][:]
        self.y = self.diafile.variables['lat_psi'][:]

        ## Labels of the terms of the M2 balance in the XI-component (in TeX code).
        self.xi_labels['ut'] = ur'$\bar{u}_t$'
        self.xi_labels['uux'] = ur'$\bar{u}\bar{u}_x$'
        self.xi_labels['vuy'] = ur'$\bar{v}\bar{u}_y$'
        self.xi_labels['ucor'] = ur'$-f\bar{v}$'
        self.xi_labels['upgrd'] = ur'$-p_x/\rho_0$'
        self.xi_labels['uistr'] = ur'$A_H$\nabla\bar{u}'
        self.xi_labels['usstr'] = ur'$\tau_s^x/\rho_0$'
        self.xi_labels['ubstr'] = ur'$-\tau_b^x/\rho_0$'
        ## Labels of the terms of the M2 balance in the ETA-component (in TeX code).
        self.eta_labels['vt'] = ur'$\bar{v}_t$'
        self.eta_labels['uvx'] = ur'$\bar{u}\bar{v}_x$'
        self.eta_labels['vvy'] = ur'$\bar{v}\bar{v}_y$'
        self.eta_labels['vcor'] = ur'$f\bar{u}$'
        self.eta_labels['vpgrd'] = ur'$-p_y/\rho_0$'
        self.eta_labels['vistr'] = ur'$A_H$\nabla\bar{v}'
        self.eta_labels['vsstr'] = ur'$\tau_s^y/\rho_0$'
        self.eta_labels['vbstr'] = ur'$-\tau_b^y/\rho_0$'

    def run_average(self, verbose=True):
        """
        USAGE
        -----
        m2_terms.run_average(verbose=True)

        Takes the time average of all terms over
        the entire run. Use CAUTION with very large
        records to avoid a MemoryError.
        """
        if self.RUN_AVERAGED:
            print "Terms have already been run-averaged."
            return
        else:
            print "Averaging %s records together."%self.xi['ut'].shape[0]
            for term in self.xi.iterkeys():
                if verbose:
                    print "Run-averaging %s term."%term
                self.xi[term] = self.xi[term][:].mean(axis=0)
            for term in self.eta.iterkeys():
                if verbose:
                    print "Run-averaging %s term."%term
                self.eta[term] = self.eta[term][:].mean(axis=0)
            self.RUN_AVERAGED = True

    def interp2line(self, ipts):
        """
        USAGE
        -----
        m2_terms.interp2line(ipts)

        Interpolates the terms to a given line with coordinates 'ipts',
        where ipts is a tuple like (lons.ravel(),lats.ravel()). Use CAUTION
        with very large records to avoid a MemoryError.
        """
        pts = (self.x.ravel(),self.y.ravel())
        if self.RUN_AVERAGED:
            for term in self.xi.iterkeys():
                print "Interpolating %s term."%term
                self.xi[term] = spint.griddata(pts, self.xi[term].ravel(), ipts, method='linear')
            for term in self.eta.iterkeys():
                print "Interpolating %s term."%term
                self.eta[term] = spint.griddata(pts, self.eta[term].ravel(), ipts, method='linear')
        else:
            print "Not implemented yet."

### CLASS PlotROMS #####################################################

class PlotROMS(object):
    """ 
    Visualization of ROMS outputs
    DEPENDS ON EXTERNAL FUNCTIONS AND CLASSES FROM ROMSLAB MODULE:
        get_depths
    !!! Under construction !!!
    """
    # getting ROMS varibles as class attributes
    def __init__(self, outname):
        self.outname = outname  
        self.outfile = nc.Dataset(outname, mode='r')
        
        varlist  = ['lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v',
                    'h', 'angle', 'ocean_time', 'temp', 'salt', 'ubar', 'vbar',
                    'u', 'v', 'zeta']
        namelist = ['lonr', 'latr', 'lonu', 'latu', 'lonv', 'latv', 'h',
                    'angle', 'time', 'temp', 'salt', 'ubar', 'vbar',
                    'u', 'v', 'zeta']
        
        for name, var in zip(namelist, varlist):
            try:
                exec "self.%s = self.outfile.variables['%s']" %(name, var)
            except KeyError:
                print "WARNING: ROMS output NetCDF file must contain the \
variable '%s' !! \n None was assined to this attribute. Some methods may \
not work properly.\n" %var
                exec "self.%s = None" %name
        
        self.lonr = self.lonr[:]; self.lonu = self.lonu[:]; self.lonv = self.lonv[:];
        self.latr = self.latr[:]; self.latu = self.latu[:]; self.latv = self.latv[:];
        self.h    = self.h[:]
        self.lm, self.km, self.im, self.jm = self.temp[:].shape
        
        dates = []
        # dates are obtained by ROMS defaut origin (0001/01/01)
        for k in range(0,len(self.time[:])):
            sec = dt.timedelta(seconds=int(self.time[k] ) )
            dates.append( dt.datetime(1, 1, 1) + sec )
        
        self.dates = dates

  
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
        

    def inimap(self, lims=None):
        """
        Initializes a Basemap object instance to use as map 
        projection for future use inside this class
        INPUT:
            lims: Map lon, lat limits [tuple: (lonmin, lonmax, latmin, latmax)]
                  If not provided, grid corners will be used as limits
        OUTPUT: Basemap object instance
        """
        if not lims:
            lims = self.corners()
        m = Basemap(projection='merc', llcrnrlon=lims[0], urcrnrlon=lims[1],
                    llcrnrlat=lims[2], urcrnrlat=lims[3], lat_ts=0, resolution='i')
        
        self.mlon, self.mlat = m(self.lonr, self.latr)
        
        return m
        
    
    def hslice(self, m, l=-1, nk=-1, velkey=1, trkey=1, vsc=10, vskip=1,
                        tr='temp', trmin=None, trmax=None):
        """
        Vel over temp or vel over salt at any Z level
        !!! Under construction !!!
        USAGE: hslice= (m, args)
        INPUT:
            m:      Basemap instance [use the method inimap to create that]
            l:      model time step [integer]
            nk:     depth [integer, meters] (Default = -1: the shalowest level  )
            velkey: flag to plot [1] or not [0] velocity vectors
            trkey:  flag to plot [1] or not [0] tracer field
            vsc:    quiver scale [integer]
            vskip:  vector skip for nice quiver presentation [integer]
            tr:     kind of tracer [string: 'temp' or 'salt']
            trmin:  minimum value for tracer colormap 
            trmax:  maximum value for tracer colormap
        OUTPUT: 
            Matplotlib figure on screen
            Figure properties and plotted arrays will become available 
            as PlotROMS class attributes
        """
        
        # preparing velocity vetors
        if velkey:
            zu = get_depths(self.outfile, l, 'u')
            zv = get_depths(self.outfile, l, 'v')
            u = 0*self.lonu; v = 0*self.lonv # initializing arrays
            if nk == -1: 
                print "\n\nSurface level was chosen! %s\n\n" % ("."*50)
                u = self.u[l, nk,...]
                v = self.v[l, nk,...]           
            else:
                print "\n\nInterpolating VEL from S to Z coordinates %s\n\n" % ("."*50)
                for a in range (0, self.im):
                    for b in range(0, self.jm-1):
                        u[a,b] = np.interp(-nk, zu[:, a, b], self.u[l, :, a, b] )
                for a in range (0, self.im-1):
                    for b in range(0, self.jm):
                        v[a,b] = np.interp(-nk, zv[:, a, b], self.v[l, :, a, b] )

            print "\n\nInterpolating (u,v) to rho-points %s\n\n" % ("."*50)
            u = griddata(self.lonu.ravel(), self.latu.ravel(), u.ravel(),
                         self.lonr, self.latr)
            v = griddata(self.lonv.ravel(), self.latv.ravel(), v.ravel(),
                         self.lonr, self.latr)
            # rotating vel according to grid angle
            u = u*np.cos(self.angle[:]) - v*np.sin(self.angle[:])
            v = u*np.sin(self.angle[:]) + v*np.cos(self.angle[:])
            
        # preparing tracer field
        if trkey:
            exec "tmp = self.%s" % (tr)
            zt   = get_depths(self.outfile, l, 'temp')
            tracer = self.lonr*0 # initializing array
            if nk == -1:
                print "\n\nSurface level was chosen! %s\n\n" % ("."*50)
                tracer = tmp[l, nk,...]
            else:
                print "\n\nInterpolating TRACER from S to Z coordinates %s\n\n" % ("."*50)
                for a in range (0, self.im):
                    for b in range(0, self.jm):
                        tracer[a,b] = np.interp(-nk, zt[:, a, b], tmp[l, :, a, b] )
        
        # masking out absurd values and in land values
        if velkey:
            u = np.ma.masked_where(u > 10, u)
            v = np.ma.masked_where(v > 10, v)
            u = np.ma.masked_where(self.h < nk, u)
            v = np.ma.masked_where(self.h < nk, v)
        if trkey:
            tracer = np.ma.masked_where(tracer > 40, tracer)
            tracer = np.ma.masked_where(self.h < nk, tracer)
        
        # initializing figure instance
        self.figure = plt.figure(facecolor='w')
        titlestr = ''
        
        if trkey:
            titlestr += "%s " %(tr)
            self.pcolor = m.pcolormesh(self.mlon, self.mlat, tracer, vmin=trmin, vmax=trmax)
            self.cbar = plt.colorbar()
            self.tracerAtZ = tracer
        if velkey:
            self.uAtZ = u
            self.vAtZ = v
            titlestr += "vel "
            self.quiver = m.quiver(self.mlon[::vskip,::vskip], self.mlat[::vskip,::vskip],
                         u[::vskip,::vskip], v[::vskip,::vskip], scale=vsc)
            vmax = np.sqrt(u**2 + v**2).max()
            self.qkey = plt.quiverkey(self.quiver, 0.6, -0.05, vmax, r"$ %.0f cm s^{-1}$" %(vmax*100),
                            labelpos='W', fontproperties={'weight': 'bold'})
                            
        self.contour = m.contour(self.mlon, self.mlat, self.h, (1000,200), colors='k')
        self.continents = m.fillcontinents()
        self.coast = m.drawcoastlines()
        titlestr += ": %04d/%02d/%02d %02d:%02dh" %(self.dates[l].year,
                     self.dates[l].month, self.dates[l].day, self.dates[l].hour,
                     self.dates[l].minute )
        if nk == -1:
            titlestr += " : Surface"
        else:
            titlestr += " : %s m" %str(nk)
        self.figTitle = plt.title(titlestr, fontsize=10, fontweight='bold')
        plt.show()
                    
        return 
        
    def vslice(self, p1, p2, sc, zlim, field, cmap=plt.cm.jet, l=-1):
        """
        Tracer vertical slice at any location
        !!! Under construction !!!
        USAGE: vslice = (p1, p2, sc, zlim, field, *args)
        INPUT:
            p1:    starting (lon,lat) point of the transect [tuple]
            p2:    ending (lon,lat) point of the transect [tuple]
            sc:   scale for contour plot (min, max, step) [tuple, list or numpy array]
            zlim:  vertical axis limits [tuple: (zmin, zmax)]
            field: field to be plotted [string: 'temp', 'salt', 'u', 'v']
            cmap:  colormap to be used
            l:     model time step [integer]
        OUTPUT:
            Matplotlib figure on screen
            Figure properties and plotted arrays will become available 
            as PlotROMS class attributes
        """
        x = self.lonr
        y = self.latr
        t = self.temp[l,...]
        s = self.salt[l,...]
        u = self.u[l,...]
        v = self.v[l,...]
        
        ths = self.outfile.variables['theta_s'][:]
        thb = self.outfile.variables['theta_b'][:]
        hc = self.outfile.variables['hc'][:]
        z = zlevs(self.h[:], self.zeta[l,...], ths, thb, hc, self.km, 'r')
        
        res = ( np.gradient(self.lonr)[1].mean() + np.gradient(self.latr)[0].mean() ) / 2
        siz = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) / res
        xs = np.linspace(p1[0], p2[0], siz)
        ys = np.linspace(p1[1], p2[1], siz)
        

        zs, ts, ss, us, vs = [], [], [], [], []
        for k in range(0, xs.size):
            lin, col = near2d( x, y, xs[k], ys[k] )
            zs.append( z[:, lin, col] )
            ts.append( t[:, lin, col] )
            ss.append( t[:, lin, col] )
            us.append( u[:, lin, col] )
            vs.append( v[:, lin, col] )
            
        zs = np.array(zs)
        ts = np.array(ts)
        ss = np.array(ss)
        us = np.array(us)
        vs = np.array(vs)
        xs.shape = (xs.size, 1)
        ys.shape = (ys.size, 1) 
        xs = xs.repeat(self.km, axis=1)
        ys = ys.repeat(self.km, axis=1)
        
        # computing cross and along transect velocity components
        
        
        exec "ps = p%s" %field[0]
        
        self.xVslice = xs
        self.yVslice = ys
        self.zVslice = zs
        self.tVslice = ts
        self.sVslice = ss
        self.uVslice = us
        self.vVslice = vs
    
        titlestr = "%s " %(tr)
        titlestr += ": %04d/%02d/%02d %02d:%02dh" %(self.dates[l].year,
                        self.dates[l].month, self.dates[l].day, self.dates[l].hour,
                        self.dates[l].minute )
        
        self.figure = plt.figure(facecolor='w')
        ax1 = self.figure.add_axes([0.1, 0.1, 0.3, 0.3])
        self.contourf = plt.contourf(xs, zs, ps, sc, axes=ax1, cmap=cmap)
        if xs[0,0] > xs[-1,-1]: ax1.set_xlim(ax1.get_xlim()[::-1])
        if p1[0] == p2[0]:
            ax1.xaxis.set_ticklabels('')
            tit = "Longitude = %s" %str(p1[0])
        else: tit = "Longitude"
        ax1.set_xlabel(tit)
        ax1.set_ylim(zlim)
        ax1.set_ylabel('z [m]')

        ax2 = self.figure.add_axes([0.5, 0.5, 0.3, 0.3])
        con = plt.contourf(ys, zs, ps, sc, axes=ax2, cmap=cmap)
        if ys[0,0] > ys[-1,-1]: ax2.set_xlim(ax2.get_xlim()[::-1])
        if p1[1] == p2[1]:
            ax2.xaxis.set_ticklabels('')
            tit = "Latitude = %s" %str(p1[1])
        else: tit = "Latitude"
        ax2.set_yticklabels('')
        ax2.set_ylim(zlim)
        ax2.xaxis.set_ticks_position('top')

        ax1.set_position( [0.125, 0.1, 0.7, 0.75] )
        ax2.set_position( [0.125, 0.1, 0.7, 0.75] )

        self.figTitle = ax1.set_title(titlestr, fontsize=10, fontweight='bold')
        self.figTitle.set_position( (0.5, 1.12) )
                
        ax3 = self.figure.add_axes([0.85, 0.1, 0.015, 0.75])
        cbar = plt.colorbar(con, cax=ax3, orientation='vertical')
        cbar.set_label('$^\circ$ C')   
            
        ax4 = self.figure.add_axes([0.4, 0.83, 0.2, 0.05]) 
        ax4.set_title(tit, fontsize=10)
        ax4.set_axis_off()

        plt.show()     
        
        return

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
    
    return z, dz            
    



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

### FUNCTION SIGMATOZ ###################################################

# def sigmatoz()

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
    #   Compute the pm and pn factors of a grid netcdf file 
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

    dat = nc.Dataset(toponame)
    x   = dat.variables['lon'][0,:]
    y   = dat.variables['lat'][:,0]
    z   = dat.variables['topo'][:]

    dxt  = np.mean(np.abs(np.diff(np.ravel(x))))
    dyt  = np.mean(np.abs(np.diff(np.ravel(y))))
    dxt  = np.mean([dxt, dyt])

    x, y = np.meshgrid(x, y)

    print '        Slicing topog data into ROMS domain'
    x, y, z = subset(x, y, z, lon.min()-1, lon.max()+1, lat.min()-1, lat.max()+1)

    # # slicing topog into roms domain
    # xm = np.ma.masked_where(x <= lon.min()-1, x)
    # ym = np.ma.masked_where(x <= lon.min()-1, y)
    # zm = np.ma.masked_where(x <= lon.min()-1, z)
    # x  = np.ma.compress_cols(xm)
    # y  = np.ma.compress_cols(ym)
    # z  = np.ma.compress_cols(zm)

    # del xm, ym, zm

    # xm = np.ma.masked_where(x >= lon.max()+1, x)
    # ym = np.ma.masked_where(x >= lon.max()+1, y)
    # zm = np.ma.masked_where(x >= lon.max()+1, z)
    # x  = np.ma.compress_cols(xm)
    # y  = np.ma.compress_cols(ym)
    # z  = np.ma.compress_cols(zm)

    # del xm, ym, zm

    # xm = np.ma.masked_where(y <= lat.min()-1, x)
    # ym = np.ma.masked_where(y <= lat.min()-1, y)
    # zm = np.ma.masked_where(y <= lat.min()-1, z)
    # x  = np.ma.compress_rows(xm)
    # y  = np.ma.compress_rows(ym)
    # z  = np.ma.compress_rows(zm)

    # del xm, ym, zm

    # xm = np.ma.masked_where(y >= lat.max()+1, x)
    # ym = np.ma.masked_where(y >= lat.max()+1, y)
    # zm = np.ma.masked_where(y >= lat.max()+1, z)
    # x  = np.ma.compress_rows(xm)
    # y  = np.ma.compress_rows(ym)
    # z  = np.ma.compress_rows(zm)

    # del xm, ym, zm

    dxr = np.mean( 1/pm )
    dyr = np.mean( 1/pn )
    dxr = np.mean([dxr, dyr])
    dxr = np.floor(dxr/1852); dxr = dxr/60

    # degrading original topog resolution according to roms
    # grid resolution to avoid unecessary heavy computations 
     
    d  = int(np.floor( dxr/dxt )) 
    if d == 0:
        d = 1
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

def get_depths(fname,tindex,type):
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
        

    try:
        h    = fname.variables['h'][:]
    except KeyError:
        h    = fname.variables['dep'][:]

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

    #   Build a smoothing coefficient that is a linear function 
    #   of a smooth topography.

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
    
    
def brunt_vaissala(rho, depth):
    """
    Computes Brunt-Vaisalla frequency
    n2 = brunt_vaissala(rho)
        rho: rho profile [1D-array] 
        depth: depth [1D-array]
    """    
    drho = np.gradient(rho)
    dz   = np.gradient(depth)
    g    = 9.8
    rho0 = 1024
    N2   = (g / rho0) * (drho / dz)
    return N2


def burger(N2, H, f, R):
    """
    Computes Burger Number based on the ratio between baroclinic deformation
    radius and curvature radius of a promontory
    USAGE: Bu = burger(N2, H, f, R)
    INPUT:
        N2: brunt vaissalla frequency based on a mean rho profile of the jet
        H:  undisturbed water depth
        f:  coriolis parameter
        R:  radius of curvature of the promontory
    """
    Bu = (N2*H**2) / (f**2 * R**2)   
    return Bu
                                                                

def rx1(z_w, rmask):
    """
    function rx1 = rx1(z_w,rmask)
 
    This function computes the bathymetry slope from a SCRUM NetCDF file.

    On Input:
       z_w         layer depth.
       rmask       Land/Sea masking at RHO-points.
 
    On Output:
       rx1         Haney stiffness ratios.
    """

    N, Lp, Mp = z_w.shape
    L=Lp-1
    M=Mp-1

    #  Land/Sea mask on U-points.
    umask = np.zeros((L,Mp))
    for j in range(Mp):
        for i in range(1,Lp):
            umask[i-1,j] = rmask[i,j] * rmask[i-1,j]

    #  Land/Sea mask on V-points.
    vmask = np.zeros((Lp,M))
    for j in range(1,Mp):
        for i in range(Lp):
            vmask[i,j-1] = rmask[i,j] * rmask[i,j-1]

    #-------------------------------------------------------------------
    #  Compute R-factor.
    #-------------------------------------------------------------------

    zx = np.zeros((N,L,Mp))
    zy = np.zeros((N,Lp,M))

    for k in range(N):
        zx[k,:] = abs((z_w[k,1:,:] - z_w[k,:-1,:] + z_w[k-1,1:,:] - z_w[k-1,:-1,:]) / 
                      (z_w[k,1:,:] + z_w[k,:-1,:] - z_w[k-1,1:,:] - z_w[k-1,:-1,:]))
        zy[k,:] = abs((z_w[k,:,1:] - z_w[k,:,:-1] + z_w[k-1,:,1:] - z_w[k-1,:,:-1]) /
                      (z_w[k,:,1:] + z_w[k,:,:-1] - z_w[k-1,:,1:] - z_w[k-1,:,:-1]))
        zx[k,:] = zx[k,:] * umask
        zy[k,:] = zy[k,:] * vmask


    r = np.maximum(np.maximum(zx[:,:,:-1],zx[:,:,1:]), np.maximum(zy[:,:-1,:],zy[:,1:,:]))

    rx1 = np.amax(r, axis=0)

    rmin = rx1.min()
    rmax = rx1.max()
    ravg = rx1.mean()
    rmed = np.median(rx1)

    print '  '
    print 'Minimum r-value = ', rmin
    print 'Maximum r-value = ', rmax
    print 'Mean    r-value = ', ravg
    print 'Median  r-value = ', rmed

    return rx1


def rx0(h, rmask):
    """
    function rx0 = rx0(h,rmask)
 
    This function computes the bathymetry slope from a SCRUM NetCDF file.

    On Input:
       h           bathymetry at RHO-points.
       rmask       Land/Sea masking at RHO-points.
 
    On Output:
       rx0         Beckmann and Haidvogel grid stiffness ratios.
    """

    Mp, Lp = h.shape
    L = Lp-1
    M = Mp-1

    #  Land/Sea mask on U-points.
    umask = np.zeros((Mp,L))
    for j in range(Mp):
        for i in range(1,Lp):
            umask[j,i-1] = rmask[j,i] * rmask[j,i-1]

    #  Land/Sea mask on V-points.
    vmask = np.zeros((M,Lp))
    for j in range(1,Mp):
        for i in range(Lp):
            vmask[j-1,i] = rmask[j,i] * rmask[j-1,i]

    #-------------------------------------------------------------------
    #  Compute R-factor.
    #-------------------------------------------------------------------

    hx = np.zeros((Mp,L))
    hy = np.zeros((M,Lp))

    hx = abs(h[:,1:] - h[:,:-1]) / (h[:,1:] + h[:,:-1])
    hy = abs(h[1:,:] - h[:-1,:]) / (h[1:,:] + h[:-1,:])

    hx = hx * umask
    hy = hy * vmask

    rx0 = np.maximum(np.maximum(hx[:-1,:],hx[1:,:]),np.maximum(hy[:,:-1],hy[:,1:]))

    rmin = rx0.min()
    rmax = rx0.max()
    ravg = rx0.mean()
    rmed = np.median(rx0)

    print '  '
    print 'Minimum r-value = ', rmin
    print 'Maximum r-value = ', rmax
    print 'Mean    r-value = ', ravg
    print 'Median  r-value = ', rmed

    return rx0


def stretching(sc, Vstretching, theta_s, theta_b):
    """
    Computes S-coordinates
    INPUT:
        sc           : normalized levels           [ndarray]
        Vstretching  : ROMS stretching algorithm   [int]
        theta_s      :                             [int] 
        theta_b      :                             [int]
        hc           :                             [int]
    """
    if Vstretching == 1:
        # Song and Haidvogel, 1994
        cff1 = 1.  / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5*theta_s)
        C = (1.-theta_b) * cff1 * np.sinh(theta_s * sc) + \
            theta_b * (cff2 * np.tanh( theta_s * (sc + 0.5) ) - 0.5)
        return C

    if Vstretching == 4:
        # A. Shchepetkin (UCLA-ROMS, 2010) double vertical stretching function
        if theta_s > 0:
            Csur = ( 1.0 - np.cosh(theta_s*sc) ) / ( np.cosh(theta_s) -1.0 )
        else:
            Csur = -sc**2

        if theta_b > 0:
            Cbot = ( np.exp(theta_b*Csur)-1.0 ) / ( 1.0-np.exp(-theta_b) )
            return Cbot
        else:
            return Csur

def get_zlev(h, sigma, hc, sc, ssh=0., Vtransform=2):
    if Vtransform == 1: # ROMS 1999
        hinv = 1./h
        cff = hc * (sc - sigma)
        if len(h.shape) > 1:
            z0 = cff[:,None,None] + sigma[:,None,None] * h[None,:,:]
        else:
            z0 = cff[:,None] + sigma[:,None] * h[None,:]
        return z0 + ssh * (1. + z0*hinv)

    elif Vtransform == 2: # ROMS 2005
        if len(h.shape) > 1:
            z0 = ( hc*sc[:,None,None] + sigma[:,None,None]*h[None,:,:] ) / ( h[None,:,:] + hc )
        else:
            z0 = ( hc*sc[:,None] + sigma[:,None]*h[None,:] ) / ( h[None,:] + hc )
        return ssh + (ssh + h) * z0
