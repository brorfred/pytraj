import sys

import numpy as np
import pylab as pl
from scipy.stats import nanmean
from matplotlib.colors import LogNorm, Normalize

import projmap
import partsat
from hitta import GBRY, WRY
import figpref
from njord import nasa

miv = np.ma.masked_invalid
def an(arr) : return arr[~np.isnan(arr)]

sys.path.append('/Users/bror/git/SoPaGyr/')
#import bender_liege

class Figs(partsat.DeltaField):

    def __init__(self,projname, casename="", **kwargs):
        super (Figs, self).__init__(projname,casename,**kwargs)
        #self.mp = projmap.Projmap('scb',resolution='f')

    def h5load(self):
        self.h5open('Dsst')
        self.Dsst = self.h5field[:]
        self.h5open('Dchl')
        self.Dchl = self.h5field[:]
        self.h5close()

    def meanplots(self):
        figpref.current()
        pl.close('all')

        def pcolor(fld, F, cmin, cmax, oneside=True):
            pl.figure(F)
            self.pcolor(miv(fld), oneside=oneside)
            pl.clim(cmin,cmax)

        """
        h = np.where(self.Dchl>0, self.Dchl, np.nan)
        pcolor(nanmean(h, axis=0), 1, 0, 2)
        h = np.where(self.Dchl<0, self.Dchl, np.nan)
        pcolor(-nanmean(h, axis=0), 2, 0, 2)
        
        h = np.where(self.Dsst>0, self.Dsst, np.nan)
        pcolor(nanmean(h, axis=0), 3, 0, 0.4)
        h = np.where(self.Dsst<0, self.Dsst, np.nan)
        pcolor(-nanmean(h, axis=0), 4, 0, 0.4)
        """
        pcolor(nanmean(self.Dchl, axis=0), 5, -0.25, 0.25, False)
        pl.title(r"Mean change of Chl (mg m$^{-3}$)")
        pl.savefig('figs/liege/meanmap_Dchl.pdf',transparent=True)
        pcolor(nanmean(self.Dsst, axis=0), 6, -0.25, 0.25, False)
        pl.title(r"Mean change of SST ($\degree$C)")
        pl.savefig('figs/liege/meanmap_Dsst.pdf',transparent=True)

    def freqmaps(self):

        figpref.current()
        pl.close(1)
        pl.figure(1)
        h = np.where(self.Dchl>0.5, 1, 0)
        fld = np.sum(h, axis=0).astype(np.float) / np.nansum(self.Dchl*0+1,0)
        self.pcolor(miv(fld*100), oneside=True)
        pl.clim(0,25)
        pl.title(r'Percent observations where Dchl > 0.5 mg $m^{-3}$')
        pl.savefig('figs/liege/freqmap_pos_Dchl.pdf',transparent=True)

        pl.close(2)
        pl.figure(2)
        h = np.where(self.Dchl<-0.5, 1, 0)
        fld = np.sum(h, axis=0).astype(np.float) / np.nansum(self.Dchl*0+1,0)
        self.pcolor(miv(fld*100), oneside=True)
        pl.clim(0,25)
        pl.title('Percent observations where Dchl < -0.5 mg $m^{-3}$')
        pl.savefig('figs/liege/freqmap_neg_Dchl.pdf')

        figpref.current()
        pl.close(3)
        pl.figure(3)
        h = np.where(self.Dsst>0.4, 1, 0)
        fld = np.sum(h, axis=0).astype(np.float) / np.nansum(self.Dsst*0+1,0)
        self.pcolor(miv(fld*100), oneside=True)
        pl.clim(0,25)
        pl.title(r'Percent observations where Dsst > 0.4 $\degree$C')
        pl.savefig('figs/liege/freqmap_pos_Dsst.pdf')

        pl.close(4)
        pl.figure(4)
        h = np.where(self.Dsst<-0.4, 1, 0)
        fld = np.sum(h, axis=0).astype(np.float) / np.nansum(self.Dsst*0+1,0)
        self.pcolor(miv(fld*100), oneside=True)
        pl.clim(0,25)
        pl.title('Percent observations where Dsst < -0.4 $\degree$C')
        pl.savefig('figs/liege/freqmap_neg_Dsst.pdf')


    def cumcumtot(self):

        figpref.current()
        pl.close(1)
        pl.figure(1)

        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                lw=1, ls=":", label='Random values, rectangular distr.')
        pl.plot(*self.cumcum(abs(np.random.randn(500000))),
                c='r', lw=1, ls=":", label='Random values, normal distr.')
        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                 lw=1, ls=":", label='Random values log-normal distr.')
    
        vec = self.Dsst[~np.isnan(self.Dsst)]
        pl.plot(*self.cumcum(vec[vec>0]),  lw=2, alpha=0.5,
                label="SST, positive values")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/cumcumtot_pos_sst.pdf',
                   transparent=True, bbox_inches='tight')
        pl.plot(*self.cumcum(-vec[vec<0]), lw=2, alpha=0.5,
                label="SST, negative values")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/cumcumtot_posneg_sst.pdf',
                   transparent=True, bbox_inches='tight')

        vec = self.Dchl[~np.isnan(self.Dchl)]
        pl.plot(*self.cumcum(vec[vec>0]),  lw=2, alpha=0.5,
                label="Chl, positive values")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/cumcumtot_posneg_sst_pos_chl.pdf',
                   transparent=True, bbox_inches='tight')

        pl.plot(*self.cumcum(-vec[vec<0]), lw=2, alpha=0.5,
                label="Chl, negative values")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/cumcumtot_posneg_sst_posneg_chl.pdf',
                   transparent=True, bbox_inches='tight')
   
        pl.savefig('figs/liege/cumcumtot.pdf')

    def cumcumtime(self):

        figpref.current()

        h5fld = self.Dsst
        tlen,_,_ = h5fld.shape
        ppp = []
        for t in np.arange(tlen):
            vec = h5fld[t,:,:]
            vec = vec[vec>0]
            if len(vec) > 0: ppp.append(self.cumpos(vec))
        #pl.close(1)
        #pl.figure(1)
        pl.plot(ppp)

    def cumcummap(self):

        figpref.current()

        h5fld = self.Dchl
        tlen,jmat,imat = h5fld.shape
        ppp = np.zeros((jmat,imat)) * np.nan
        for j in np.arange(jmat):
            for i in np.arange(imat):
                vec = h5fld[:,j,i]
                vec = vec[vec>0]
                if len(vec) > 0: ppp[j,i] = self.cumpos(vec)
        pl.close(1)
        pl.figure(1)
        self.pcolor(miv(ppp), oneside=True)
        pl.clim(0.3,0.8)
        pl.title(r'Cumulative Dchl for 10% highest pos obs.')
        pl.savefig('figs/liege/cumcummap_pos_Dchl.pdf')

        
        ppp = np.zeros((jmat,imat)) * np.nan
        for j in np.arange(jmat):
            for i in np.arange(imat):
                vec = h5fld[:,j,i]
                vec = -vec[vec<0]
                if len(vec) > 0: ppp[j,i] = self.cumpos(vec)
        pl.close(2)
        pl.figure(2)
        self.pcolor(miv(ppp), oneside=True)
        pl.clim(0.3,0.8)
        pl.title(r'Cumulative Dchl for 10% highest neg obs.')
        pl.savefig('figs/liege/cumcummap_neg_Dchl.pdf')

        h5fld = self.Dsst

        tlen,jmat,imat = h5fld.shape
        ppp = np.zeros((jmat,imat)) * np.nan
        for j in np.arange(jmat):
            for i in np.arange(imat):
                vec = h5fld[:,j,i]
                vec = vec[vec>0]
                if len(vec) > 0: ppp[j,i] = self.cumpos(vec)
        pl.close(3)
        pl.figure(3)
        self.pcolor(miv(ppp), oneside=True)
        pl.clim(0.3,0.8)
        pl.title(r'Cumulative Dsst for 10% highest pos obs.')
        pl.savefig('figs/liege/cumcummap_pos_Dsst.pdf')

        ppp = np.zeros((jmat,imat)) * np.nan
        for j in np.arange(jmat):
            for i in np.arange(imat):
                vec = h5fld[:,j,i]
                vec = -vec[vec<0]
                if len(vec) > 0: ppp[j,i] = self.cumpos(vec)
        pl.close(4)
        pl.figure(4)
        self.pcolor(miv(ppp), oneside=True)
        pl.clim(0.3,0.8)
        pl.title(r'Cumulative Dsst for 10% highest neg obs.')
        pl.savefig('figs/liege/cumcummap_neg_Dsst.pdf')

    def histmoeller(self, fldname="Dchl"):
        """Create a hofmoeller representation of diff distributions"""
        self.jdvec = np.arange(self.jdmin,self.jdmax)
        if fldname is "Dchl":
            self.hpos = np.linspace(0, 20, 100)
            mat = self.Dchl
        elif fldname is "Dsst":
            self.hpos = np.linspace(0, 5, 100)
            mat = self.Dsst

        self.posmat = np.zeros((len(self.jdvec), len(self.hpos)-1))
        self.negmat = np.zeros((len(self.jdvec), len(self.hpos)-1))
        self.posmean = []
        self.negmean = []
        for n,jd in enumerate(self.jdvec):
            diff = mat[n,:,:]
            diff[diff==0] = np.nan
            self.negmat[n,:],_ = np.histogram(-diff[diff<0], self.hpos)
            self.posmat[n,:],_ = np.histogram(diff[diff>0], self.hpos)
            self.posmean.append(np.mean(diff[diff>0]))
            self.negmean.append(np.mean(diff[diff<0]))
            print n,jd,len(-diff[diff<0]),len(diff[diff>0]),self.posmat.max()
        self.posmean = np.array(self.posmean)
        self.negmean = np.array(self.negmean)

    def histmoellerplot(self, fldname="Dchl"):

        self.histmoeller(fldname)
        figpref.current()
        pl.close(1)
        fig = pl.figure(1)
        pl.subplots_adjust(hspace=0,right=0.85, left=0.15)

        def subplot(sp, mat, vec, xvec):
            ax = pl.subplot(2,1,sp, axisbg='0.8')
            im =pl.pcolormesh(self.jdvec, xvec, miv(mat.T), rasterized=True,
                              norm=LogNorm(vmin=1,vmax=10000), cmap=WRY())
            pl.plot(self.jdvec, vec, lw=2)
            pl.gca().xaxis.axis_date()
            pl.setp(ax, xticklabels=[])
            return im

        im = subplot(1, self.posmat, self.posmean,  self.hpos)
        im = subplot(2, self.negmat, self.negmean, -self.hpos)

        pl.figtext(0.08,0.5, r'%s d$^{-1}$)' % fldname,
                   rotation='vertical', size=16, va='center')
        cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label('Number of grid cells')
        pl.savefig('figs/liege/histmoeller_%s.pdf' % fldname)

        pl.close(2)
        fig = pl.figure(2)
        pl.fill_between(self.jdvec, self.jdvec*0, self.posmean,
                        facecolor='powderblue')
        pl.fill_between(self.jdvec, self.negmean, self.jdvec*0,
                        facecolor='pink')
        pl.plot(self.jdvec, self.negmean+self.posmean,lw=1,c='k')
        #pl.title(fldname)
        pl.gca().xaxis.axis_date()

        pl.savefig('figs/liege/meantime_%s.pdf' % fldname)

    def meantime2plot(self, sst=True, chl=True):

        figpref.current()
        pl.close(1)
        fig = pl.figure(1)
        pl.subplots_adjust(hspace=0,right=0.85, left=0.15)

        def plot(fldname):
            self.histmoeller(fldname)            
            pl.fill_between(self.jdvec, self.jdvec*0, self.posmean,
                            facecolor='powderblue')
            pl.fill_between(self.jdvec, self.negmean, self.jdvec*0,
                            facecolor='pink')
            pl.plot(self.jdvec, self.negmean+self.posmean,lw=1,c='k')
            pl.gca().xaxis.axis_date()


        if sst:
            ax = pl.subplot(2,1,1)
            plot('Dsst')
            pl.ylabel('Dsst ($\degree$C d$^{-1}$')
            pl.setp(ax.get_xticklabels(), visible=False)        
        if chl:
            ax = pl.subplot(2,1,2)
            plot('Dchl')
            pl.ylim(-2,2)
            pl.yticks([-2,-1,0,1,2])
            pl.ylabel('Chl (mg m$^{3}$ d$^{-1}$)')
        pl.savefig('figs/liege/meantime_both.pdf')


    def all(self):
        self.h5load()
        self.cumcummap()
        self.cumcumtot()
        self.meanplots()
        self.freqmaps()
        self.histmoellerplot('Dchl')
        self.histmoellerplot('Dsst')


    def globcumcum(self):
        """Create global cucum curves from MODIS 4km data"""
        ns = nasa.MODIS(res='4km')
        jd1 = pl.datestr2num('2003-01-01')
        jd2 = pl.datestr2num('2012-12-31')

        self.northmat = np.zeros((100,jd2-jd1)) 
        self.globmat = np.zeros((100,jd2-jd1))
        self.southmat = np.zeros((100,jd2-jd1)) 

        for n,jd in enumerate(np.arange(jd1,jd2)):
            ns.load('chl',jd=jd)
            _,y = self.cumcum(an(ns.chl))
            self.globmat[:,n] = y
            _,y = self.cumcum(an(ns.chl[:ns.jmt/2,:]))
            self.northmat[:,n] = y
            _,y = self.cumcum(an(ns.chl[ns.jmt/2:,:]))
            self.southmat[:,n] = y
            print jd2-jd


    def cumcum_theory(self):

        figpref.current()
        pl.close(1)
        pl.figure(1)
        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                    lw=2, ls="-", label='Random values, rectangular distr.')
        pl.legend(loc='lower right')
        pl.ylim(0,1)
        pl.savefig('figs/liege/cc_theory_rectdist.pdf',
                   transparent=True, bbox_inches='tight')

        pl.plot(*self.cumcum(abs(np.random.randn(500000))), c='r',
                lw=2, ls="-", label='Random values, normal distr.')
        pl.legend(loc='lower right')
        pl.ylim(0,1)
        pl.savefig('figs/liege/cc_theory_normrectdist.pdf',
                   transparent=True, bbox_inches='tight')

        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                lw=2, ls="-", label='Random values log-normal distr.')   
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/cc_theory_lognormrectdist.pdf',
                   transparent=True, bbox_inches='tight')



    def example_select(self,field="chl", jd=734107):
        """ Get the interpolated tracer for all trajectories at jd=jd"""
        dtype = np.dtype([('runid',np.int), ('ntrac',np.int), ('jd',np.float),
                          ('t1',np.float), ('t2',np.float),
                          ('val1',np.float), ('val2',np.float),
                          ('x',np.float),  ('y',np.float)])
        sql = """SELECT t.runid, t.ntrac, t.ints, c.ints as t1,
                        c2.ints as t2, c.val  as val1, c2.val  as val2,
                        t.x, t.y FROM %s__%s c
                    INNER JOIN %s__%s c2 ON
                            c.runid=c2.runid AND c.ntrac=c2.ntrac
                    INNER JOIN %s t ON c.runid=t.runid AND c.ntrac=t.ntrac
                  WHERE c.ints > %i AND  c.ints <= %i AND
                          c2.ints > %i AND c2.ints < %i AND t.ints=%i;"""
        self.c.execute(sql % (self.tablename, field, self.tablename, field,
                              self.tablename, jd-10, jd, jd, jd+10, jd))
        self.res = np.array(self.c.fetchall(), dtype=dtype)
        if len(self.res)==0: return False
        dt =  (self.res['t2'] - self.res['t1'])
        r1 = (self.res['t2'] - jd) / dt
        r2 = (jd - self.res['t1']) / dt
        setattr(self, field+'vec', self.res['val2']*r1 + self.res['val1'] * r2)
        self.x = self.res['x']
        self.y = self.res['y']
        self.jd = self.res['jd']
        self.ntrac = self.res['ntrac']
        self.runid = self.res['runid']

    def interp_example(self, x=50, y=50, jd=734107):

        self.example_select(jd=jd)
        mask = ((self.x.astype(np.int)==x) & (self.y.astype(np.int)==y) &
                (self.jd==jd))
        dtype = np.dtype([('runid',np.int), ('jd',np.float), ('ntrac',np.int),
                          ('x',np.float),  ('y',np.float),  ('z',np.float)])
        sql = ("SELECT * FROM %s WHERE ntrac=%i AND ints>%f " +
               " AND ints<%f AND runid=%i")
        pl.clf()
        for runid, ntrac in zip(self.runid[mask], self.ntrac[mask]):
            self.c.execute( sql % (self.tablename, ntrac, jd-10, jd+10, runid))
            res = np.array(self.c.fetchall(), dtype=dtype)
            pl.plot(res['x'], res['y'])


    def glob(self):

        mat = np.load('globcummat.npz')

        figpref.current()
        pl.close(1)
        pl.figure(1)

        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                lw=1, ls=":", label='Random values, rectangular distr.')
        pl.plot(*self.cumcum(abs(np.random.randn(500000))),
                c='r', lw=1, ls=":", label='Random values, normal distr.')
        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                lw=1, ls=":", label='Random values log-normal distr.')


        pl.plot(np.linspace(0,1,100), np.mean(mat['globmat'],axis=1),
                lw=2, alpha=0.5, label="Global")
        pl.plot(np.linspace(0,1,100), np.mean(mat['northmat'],axis=1),
                lw=2, alpha=0.5, label="Northern hemisphere")
        pl.plot(np.linspace(0,1,100), np.mean(mat['southmat'],axis=1),
                lw=2, alpha=0.5, label="Southern hemisphere")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/glob_glob.pdf',
                   transparent=True, bbox_inches='tight')

    def globtime(self):

        mat = np.load('globcummat.npz')
        jd1 = pl.datestr2num('2003-01-01')
        jd2 = pl.datestr2num('2012-12-31')
        jdvec = np.arange(jd1,jd2)
        
        figpref.current()
        pl.close(1)
        pl.figure(1)
        
        pl.plot_date(jdvec, mat['northmat'][10,:],'r-',
                     alpha=0.5, label="Northern hemisphere")
        pl.plot_date(jdvec, mat['southmat'][10,:],'b-',
                     alpha=0.5, label="Southern hemisphere")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/glob_time.pdf',
                   transparent=True, bbox_inches='tight')
        

    def latslice(self, datadir='/Volumes/keronHD3/globChl/result/'):
        from scipy.io import loadmat

        figpref.current()
        pl.close(1)
        pl.figure(1)

        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                lw=1, ls=":", label='Random values, rectangular distr.')
        pl.plot(*self.cumcum(abs(np.random.randn(500000))),
                c='r', lw=1, ls=":", label='Random values, normal distr.')
        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                lw=1, ls=":", label='Random values log-normal distr.')


        for lat in [0,15,30,45,60,75]:
            mat = loadmat(datadir + 'latsliceschl_lat%i.mat' % lat)
            
            pl.plot(*self.cumcum(mat['chl']),  lw=2, alpha=0.5,
                    label="Chl, lat=%i" % lat)
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/glob_latslice.pdf',
                   transparent=True, bbox_inches='tight')

    def o2ar(self):

        cr =bender_liege.cruises.all_cruises()

        figpref.current()
        pl.close(1)
        pl.figure(1)


        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                lw=1, ls=":", label='Random values, rectangular distr.')
        pl.plot(*self.cumcum(abs(np.random.randn(500000))),
                c='r', lw=1, ls=":", label='Random values, normal distr.')
        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                lw=1, ls=":", label='Random values log-normal distr.')

        vec = cr.ncp[cr.ncp>0]
        pl.plot(*self.cumcum(vec),  lw=2, alpha=0.5,
                label="O2Ar based NCP")
        pl.ylim(0,1)
        pl.legend(loc='lower right')
        pl.savefig('figs/liege/glob_o2ar.pdf',
                   transparent=True, bbox_inches='tight')
        

def cumcum_theory():

    figpref.current()
    pl.close(1)
    pl.figure(1)
        
    #a = np.random.randn(4,4)
    a = np.random.lognormal(0,1,(4,4))
    pl.clf()
    pl.pcolormesh(abs(a), cmap=WRY())
    pl.yticks([])
    pl.xticks([])
    pl.savefig('figs/liege/cc_theory_grid.pdf',
               transparent=True, bbox_inches='tight')

    b = np.reshape(np.abs(a),(1,16))
    pl.clf()
    pl.subplot(8,1,1)
    pl.pcolormesh(b, cmap=WRY())
    pl.yticks([])
    pl.xticks([])
    pl.savefig('figs/liege/cc_theory_vec.pdf',
               transparent=True, bbox_inches='tight')

    c = np.sort(b)[:,::-1]
    pl.clf()
    pl.subplot(8,1,1)
    pl.pcolormesh(c, cmap=WRY())
    pl.yticks([])
    pl.xticks([])
    pl.savefig('figs/liege/cc_theory_sortvec.pdf',
               transparent=True, bbox_inches='tight')

    d = np.hstack((0,np.squeeze(np.cumsum(c)/np.sum(c))))
    pl.clf()
    pl.plot(np.linspace(0,1,17), d)
    pl.ylim(0,1)
    #pl.xticks([])
    pl.savefig('figs/liege/cc_theory_cumcum.pdf',
               transparent=True, bbox_inches='tight')
    
  

    
