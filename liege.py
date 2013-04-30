
import numpy as np
import pylab as pl
from scipy.stats import nanmean
from matplotlib.colors import LogNorm, Normalize

import projmap
import partsat
from hitta import GBRY, WRY
import figpref

miv = np.ma.masked_invalid

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
        pl.title("Mean change of Chl")
        pl.savefig('figs/liege/meanmap_Dchl.pdf')
        pcolor(nanmean(self.Dsst, axis=0), 6, -0.25, 0.25, False)
        pl.title("Mean change of SST")
        pl.savefig('figs/liege/meanmap_Dsst.pdf')

    def freqmaps(self):

        figpref.current()
        pl.close(1)
        pl.figure(1)
        h = np.where(self.Dchl>0.5, 1, 0)
        fld = np.sum(h, axis=0).astype(np.float) / np.nansum(self.Dchl*0+1,0)
        self.pcolor(miv(fld*100), oneside=True)
        pl.clim(0,25)
        pl.title(r'Percent observations where Dchl > 0.5 mg $m^{-3}$')
        pl.savefig('figs/liege/freqmap_pos_Dchl.pdf')

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
        pl.plot(*self.cumcum(abs(np.random.randn(500000))), c='r', lw=1, ls=":",
                label='Random values, log-normal distr.')
        pl.plot(*self.cumcum(abs(np.random.lognormal(0,1,500000))), c='b',
                 lw=1, ls=":", label='Random values normal distr.')
        pl.plot(*self.cumcum(abs(np.random.rand(500000))), c='g',
                lw=1, ls=":", label='Random values, rectangular distr.')

        vec = self.Dchl[~np.isnan(self.Dchl)]
        pl.plot(*self.cumcum(vec[vec>0]),  lw=2, alpha=0.5,
                label="Chl, positive values")
        pl.plot(*self.cumcum(-vec[vec<0]), lw=2, alpha=0.5,
                label="Chl, negative values")

        vec = self.Dsst[~np.isnan(self.Dsst)]
        pl.plot(*self.cumcum(vec[vec>0]),  lw=2, alpha=0.5,
                label="SST, positive values")
        pl.plot(*self.cumcum(-vec[vec<0]), lw=2, alpha=0.5,
                label="SST, positive values")

        pl.legend(loc='lower right')
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

        histmoeller(self, fldname)
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
        pl.plot(self.jdvec, self.negmean+self.posmean,lw=1,c='k')
        pl.plot(self.jdvec, self.posmean,':',lw=2)
        pl.plot(self.jdvec, self.negmean,':',lw=2)
        pl.title(fldname)
        pl.gca().xaxis.axis_date()

        pl.savefig('figs/liege/meantime_%s.pdf' % fldname)


    def all(self):
        self.h5load()
        self.cumcummap()
        self.cumcumtot()
        self.meanplots()
        self.freqmaps()
        self.histmoellerplot('Dchl')
        self.histmoellerplot('Dsst')

