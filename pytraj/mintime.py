import os, sys
import numpy as np
import pylab as pl
from scipy.ndimage.measurements import label
from scipy import spatial
from scipy import cluster


from njord import ecco
from njord.utils import figpref

try:
 import mycolor
 from hitta import WRY,GBRY
except ImportError, e:
 pass 

import trm
import connect
import lldist

miv = np.ma.masked_invalid

class MinTimeMatrix(connect.ConnectivityMatrix):

    def __init__(self,projname,casename="", **kwargs):
        super(Global,self).__init__(projname, casename, **kwargs)
        self.npzdir = "data"
        self.imat,self.jmat = np.meshgrid(np.arange(self.imt),
                                          np.arange(self.jmt))

    def north_atl_mask(self):
        """Create a mask for the North Atlantic region"""
        self.gcm.add_landmask()
        mask = self.gcm.landmask * False
        arr,n = label(~self.gcm.landmask[360:640,200:1000])
        mask[360:640,200:1000][arr==arr[150, 400]] = True
        if hasattr(self.gcm,'gmt'):
            return self.gcm.gmt.field(mask)
        else:
            return mask

    def global_mask(self):
        """Create a mask for the North Atlantic region"""
        self.gcm.add_landmask()
        return self.gcm.landmask

    def create_seedfile(self,reg='NA'):
        """Create seed file for the North Atlantic"""
        mask = north_atl_mask(self)
        self.create_seedfile('na_seed.asc',49,mask)

    def add_initpos(self,initfname=None):
        """Add separate vectors with initial particle postions""" 
        if initfname is None:
            filename = self.currfile()
            initfname = filename[:-14] + '000001_ini' + filename[-4:]
        self.load(filename=initfname)
        for v in ['jd','ntrac','x','y','z']:
            self.__dict__['ini' + v] = self.__dict__[v]
     
    def gen_regmat(self, xvec=None, yvec=None, mask=None, regdi=2, regdj=2):
        """Generate a region mask by aggregating grid cells"""
        self.regmat = self.imat * 0
        if (xvec is None) &  (xvec is None):
            if mask is None: mask = ~self.global_mask()
            xvec = np.ravel(self.imat[mask])
            yvec = np.ravel(self.jmat[mask])
        regvec = np.ravel_multi_index(
            np.vstack(((xvec/regdi).astype(int), (yvec/regdj).astype(int))),
            (int(self.imt/regdi), int(self.jmt/regdj))) + 1
        unvec,indices = np.unique(regvec, return_inverse=True)
        regvec = np.arange(1, len(unvec)+1)[indices]
        self.regmat[yvec.astype(np.int),xvec.astype(np.int)] = regvec
        self.nreg = len(np.unique(regvec)) + 1
        self.regdi = regdi; self.regdj = regdj

        
    def add_regvec(self):
        """Add a vector with region ID's based on a regionmask"""
        self.reg = self.regmat[self.y.astype(np.int),
                               self.x.astype(np.int)]

    def plot_timemap(self, regid):
        self.tmap = self.regmat[:] * 0.
        for n,t in enumerate(self.conmat[:,regid]): 
            self.tmap[self.regmat==n+1] = t

    def calc_mintime(self, regdi=5, regdj=5):
        """Create connectivity matrix using a regions matrix."""
        self.add_initpos()
        self.gen_regmat(self.inix, self.iniy, regdi, regdj)
        self.mintmat = np.zeros((self.nreg,self.nreg), dtype=np.float16) - 1
        ntracmax = self.inintrac.max()
        convec = np.zeros((2, ntracmax+1), dtype=np.int)
        jdmin = self.inijd.min()

        self.add_regvec()
        self.inireg = self.reg.copy()
        convec[0,self.inintrac] = self.inireg#[self.inintrac]
     
        for filename in np.sort(self.list_partfiles())[::-1]:
            print filename
            self.load(filename=filename)
            self.add_regvec()
            for jd in self.jdvec[:0:-1]:
                tmask = self.jd == jd
                convec[1,:] = 0
                convec[1,self.ntrac[tmask]] = self.reg[tmask].astype(np.int)
                fltcrds = np.ravel_multi_index(convec, (self.nreg, self.nreg))
                sums = np.bincount(fltcrds, minlength=self.nreg*self.nreg)
                self.mintmat.flat[sums>0] = np.float16(jd-jdmin)
        self.mintmat[self.mintmat==-1] = np.nan
        npzname = ("mintime_%s%s_%02i_%02i.npz" %
                   (self.projname, self.casename, regdi,regdj))
        np.savez(os.path.join('data',npzname),
                 mintmat=self.mintmat, regmat=self.regmat,
                 regdi=self.regdi, regdj=self.regdj, nreg=self.nreg,
                 llat=self.llat, llon=self.llon)

    def read_mintmat(self, regdi=5, regdj=5):
            npzname = ("mintime_%s%s_%02i_%02i.npz" %
                       (self.projname, self.casename, regdi,regdj))
            npzfile = np.load(os.path.join(self.npzdir,npzname))
            for var in ['mintmat', 'regmat', 'regdi', 'regdj', 'nreg']:
                self.__dict__[var] = npzfile[var]

    def distsorted_unvec(self):
        self.calc_reglatlon()
        lonmat,_ = np.meshgrid(self.reglonvec, self.reglonvec)
        latmat,_ = np.meshgrid(self.reglatvec, self.reglatvec)
        distvec = lldist.ll2dist2vec(np.ravel(lonmat),   np.ravel(latmat),
                                     np.ravel(lonmat.T), np.ravel(latmat.T))
        self.regdistmat = distvec.reshape(self.nreg, self.nreg)
        pvec = spatial.distance.pdist(self.regdistmat[1:,1:])
        link = cluster.hierarchy.linkage(pvec, method='average')
        return np.hstack((0, cluster.hierarchy.leaves_list(link)))
        
    """
    maxi,maxj = np.nonzero(self.regdistmat==np.nanmax(self.regdistmat))
    sprdvec = lldist.ll2dist2vec(self.reglonvec, self.reglatvec,
    lonmat[maxj[0],maxi[0]],
    latmat[maxj[0],maxi[0]])
    sprdvec[0] = -1
    return unvec[np.argsort(sprdvec)]
    """
        
    def calc_reglatlon(self):
        """Calculate the center (mean) position of each region.""" 
        self.reglatvec = (np.bincount(np.ravel(self.regmat),
                                      np.ravel(self.llat)) /
                          np.bincount(np.ravel(self.regmat)))
        self.reglonvec = (np.bincount(np.ravel(self.regmat),
                                      np.ravel(self.llon)) /
                          np.bincount(np.ravel(self.regmat)))
        self.reglatvec[0] = np.nan
        self.reglonvec[0] = np.nan
       


    def extract_trajs_by_region(self,regid, crit="from", regdi=5, regdj=5):
        """Extract all trajctories that originaties from a specific region."""
        self.add_initpos()
        self.gen_regmat(self.inix, self.iniy, regdi, regdj)
        jdmin = self.inijd.min()
        self.add_regvec()
        if crit is "from":
            ntracvec = self.inintrac[self.reg==regid]
        else:
            ntracvec = np.array([])
            for filename in np.sort(self.list_partfiles()):
                print filename
                self.load(filename=filename)
                self.add_regvec()
                ntracvec = np.append(
                    ntracvec,np.unique(self.ntrac[self.reg==regid]))
            ntracvec = np.unique(ntracvec).astype(np.int)
        
        class sub: pass
        subvars = ['ntrac','jd','x','y','z','reg']
        for v in subvars:
            sub.__dict__[v] = np.array([])
        for filename in np.sort(self.list_partfiles()):
            print filename
            self.load(filename=filename)
            self.add_regvec()
            self.mask_by_ntracvec(ntracvec, ['reg'])
            for v in subvars:
                sub.__dict__[v] = np.append(sub.__dict__[v], self.__dict__[v])
        for v in subvars:
            self.__dict__[v] = sub.__dict__[v]
        self.jdvec = np.unique(self.jd)

    def plot_numreg(self, maxdays, regdi=5, regdj=5):
        """Plot number of regions connected to each region"""
        if not hasattr(self,'mintmat'):
            self.read_mintmat(self,regdi=regdi, regdj=regdj)

        def calc_numregmat(loc="to"):
            axis = 1 if loc is "to" else 0
            numreg = self.regmat[:] * np.nan
            numvec = np.sum(np.where(self.mintmat<=maxdays, 1, 0), axis=axis)
            for r,n in zip(np.arange(1,self.nreg+1), numvec[1:]):
                numreg[self.regmat==r] = n
            return numreg
    
        numregfrom = calc_numregmat("from")
        numregto   = calc_numregmat("to")
        lim = max(np.nanmax(numregto,   axis=None),
                  np.nanmax(numregfrom, axis=None))

        figpref.current()
        pl.close(1)
        fig = pl.figure(1,(8,11))
        pl.clf()
        pl.suptitle("Connectivity after %i days" % maxdays)

        pl.subplots_adjust(hspace=0, top=0.95,bottom = 0.15)
        pl.subplot(3,1,1, axisbg="0.8")
        self.gcm.pcolor(miv(numregfrom), cmap=WRY(), rasterized=True)
        pl.clim(0,lim)
        self.gcm.mp.text(70,60,'Number of source regions')
        
        pl.subplot(3,1,2, axisbg="0.8")
        self.gcm.pcolor(miv(numregto), cmap=WRY(), rasterized=True)
        pl.clim(0,lim)
        self.gcm.mp.text(70,60,'Number of sink regions')

        pl.subplot(3,1,3, axisbg="0.8")
        self.gcm.pcolor(miv(numregto-numregfrom), cmap=GBRY(), rasterized=True)
        pl.clim(-lim,lim)
        self.gcm.mp.text(70,60,'Difference, sinks-sources')

        if 'mycolor' in sys.modules:
            mycolor.freecbar([0.2,0.12,0.6,0.025],[-lim,-lim/2,0,lim/2,lim],
                             cmap=GBRY())
        pl.savefig('data/numreg_%03i.pdf' % maxdays)

    def plot_mintmin(self, regdij):
        """ Plot the min time for connection between regions """
        fname = 'data/mintime_ecco025ecco025_%02i_%02i.npz'
        regmat  = np.load(fname % (regdij,regdij))['regmat'].astype(np.float)
        mintmat = np.load(fname % (regdij,regdij))['mintmat'].astype(np.float)
        figpref.current()
        pl.close(1)
        fig = pl.figure(1,(8,9))
        pl.subplot(2,1,1, axisbg="0.7")
        regmat[regmat==0] = np.nan
        self.gcm.pcolor(regmat)
        pl.subplot(2,1,2)
        pl.pcolormesh(miv(mintmat))
        pl.clim(0,365*3)
        pl.colorbar()
        pl.suptitle('reg size = %i gridcells' % regdij)
        pl.savefig('figs/mintmin_%03i.png' % regdij)

    def plot_onereg(self, regid):
        """Plot mintime of connectivity for one individual region"""
        tomat = self.llat * np.nan
        for i,v in enumerate(self.mintmat[1:,regid]):
            tomat[self.regmat==(i+1)] = v
        frmat = self.llat * np.nan
        for i,v in enumerate(self.mintmat[regid,1:]):
            frmat[self.regmat==(i+1)] = v
        djtk,lim = djdticks(max(np.nanmax(tomat), np.nanmax(frmat)))
        figpref.current()
        pl.close(1)
        fig = pl.figure(1,(8,8))
        pl.clf()
        pl.suptitle("Connectivity for region %i" % regid)

        mask = self.regmat == regid
        x,y = self.gcm.mp(self.llon[mask], self.llat[mask])
        pl.subplots_adjust(hspace=0, top=0.95,bottom = 0.15)

        pl.subplot(2,1,1, axisbg="0.8")
        self.gcm.pcolor(frmat, cmap=WRY(), rasterized=True)
        pl.clim(0,lim)
        self.gcm.mp.text(70,60,'Time from region')
        self.gcm.mp.scatter(x,y,5,'b')

        pl.subplot(2,1,2, axisbg="0.8")
        self.gcm.pcolor(tomat, cmap=WRY(), rasterized=True)
        pl.clim(0,lim)
        self.gcm.mp.text(70,60,'Time to region')
        self.gcm.mp.scatter(x,y,5,'b')

        if 'mycolor' in sys.modules:
            mycolor.freecbar([0.2,0.12,0.6,0.025], djtk, cmap=WRY())
        pl.savefig('figs/onereg_%02i_%02i_%06i.png' %
                   (self.regdi, self.regdj, regid))

        
def extract_trajs_through_region(self,regid):
    """Extract all trajctories that originaties from a specific region."""
    add_initpos(self)
    gen_regmat(self, self.inix, self.iniy,10,10)
    jdmin = self.inijd.min()
    add_regvec(self)
    ntracvec = np.array([])
    for filename in np.sort(self.list_partfiles()):
        print filename
        self.load(filename=filename)
        add_regvec(self)
        ntracvec = np.append(ntracvec,np.unique(self.ntrac[self.reg==regid]))
    ntracvec = np.unique(ntracvec).astype(np.int)

    class sub: pass
    subvars = ['ntrac','jd','x','y','z','reg']
    for v in subvars:
        sub.__dict__[v] = np.array([])
    for filename in np.sort(self.list_partfiles()):
        print filename
        self.load(filename=filename)
        add_regvec(self)
        mask_by_ntrac(self, ntracvec, ['reg'])
        for v in subvars:
            sub.__dict__[v] = np.append(sub.__dict__[v], self.__dict__[v])
    for v in subvars:
        self.__dict__[v] = sub.__dict__[v]
    self.jdvec = np.unique(self.jd)


def mintimemat_from_vecs(self):
    mindjmat = self.llat.copy() * np.nan
    jdmin = self.inijd.min()
    for jd in self.jdvec[:0:-1]:
        tmask = self.jd == jd
        mindjmat[self.y[tmask].astype(np.int),
                 self.x[tmask].astype(np.int)] = jd
    return mindjmat


def djdticks(maxdjd):

    maxdjd = float(maxdjd)
    if maxdjd <= 60:
        maxval = np.ceil((maxdjd/7)/2)*2
        return ['%gw' % f for f in np.linspace(0,maxval,5)], maxval*7
    if maxdjd <= 365:
        maxval = np.ceil((maxdjd/30.5)/2)*2
        return ['%gm' % f for f in np.linspace(0,maxval,5)], maxval*30.5
    maxval = np.ceil((maxdjd/365)/2)*2
    return ['%gyr' % f for f in np.linspace(0,maxval,5)], maxval*365
"""
def mintimemat_from_vecs(self):
for jd in self.jdvec[:0:-1]:
    tmask = self.jd == jd
    scatter(tr.x[tmask],tr.y[tmask],5,tr.jd[tmask])
    clim(732200,733800)
"""


