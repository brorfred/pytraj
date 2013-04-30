import datetime
from datetime import datetime as dtm
import glob
import os
from itertools import izip
import cStringIO
import subprocess as spr

import numpy as np
import pylab as pl
import matplotlib as mpl

from traj import Traj

import namelist as nlt
import lldist

"""trm - a module to post-process output from TRACMASS""

This module simplifies reading and analysis of data genereated by the TRACMASS
off-line perticle tracking code (http://tracmass.org). The module also 
includes functionality to generate seed files to be used when a specific
region should be seeded by TRACMASS. 
"""
class Trm(Traj):
    """Class for TRACMASS specific functionalily
         Usage:
           tr = pytraj.Trm('projname', 'casename')
           tr.load()                    #Load the last generated file
           tr.load(jd=1234)             #Load the file starting at 1234
           tr.load (filename"file.bin") #Load the file named file.bin   
         
         Before using this module, set the environmental variable TRM 
         to point to the directory with your TRACMASS project. Add the 
         following line to the .profile, .bashrc, or other configuration
         file in your home directory:
        
           export TRMDIR="/path/to/the/trm/root/dir/"                     
    """

    def __init__(self,projname, casename=None, **kwargs):

        self.__dict__['part'] = None
        self.__dict__['rank'] = None
        self.__dict__['arg1'] = None
        self.__dict__['arg2'] = None
        super(Trm, self).__init__(projname, casename, **kwargs)
        if not hasattr(self, 'trmdir'):
            self.trmdir = os.getenv('TRMDIR')
            if self.trmdir is None:
                raise EnvironmentError, """ Trmdir is not set.
                Add TRMDIR=/path/to/tracmass to your local environment
                or specify trmdir when calling Trm."""

        def parse(od,pn,cn,sfx):
            gridfile = '/%s/projects/%s/%s_%s.in' % (od, pn, cn, sfx)
            if not os.path.isfile(gridfile):
                raise IOError("Can't find the file %s" % gridfile)
            return nlt.parse(gridfile)
        self.nlgrid = parse(self.trmdir,self.projname,self.projname,"grid")
        self.nlrun =  parse(self.trmdir,self.projname,self.casename,"run")

        if not hasattr(self, 'datadir'): self.datadir = self.nlrun.outDataDir
        if not hasattr(self, 'datafile'): self.datafile=self.nlrun.outDataFile

        self.base_iso = pl.date2num(dtm(
            self.nlgrid.baseYear,
            self.nlgrid.baseMon,
            self.nlgrid.baseDay))-1
        self.imt = self.nlgrid.IMT
        self.jmt = self.nlgrid.JMT

        if not os.path.isabs(self.datadir):
            self.datadir = os.path.join(self.trmdir, self.datadir)

    @property
    def dtype(self):
        return np.dtype([('ntrac','i4'), ('ints','f8'), 
                         ('x','f4'), ('y','f4'), ('z','f4')])

    def read_bin(self, filename, count=-1):
        """Read binary output from TRACMASS"""
        return np.fromfile(open(filename), self.dtype, count=count)

    def read_asc(self, filename):
        """Read ascii output from TRACMASS"""
        return np.genfromtxt(filename, self.dtype)

    def read_jdrange(self, filename):
        """Read last reacord in binary file"""
        with open(os.path.join(self.datadir, filename)) as fH:
            try:
                row = np.fromfile(fH, self.dtype, 1)
                self.firstjd = row['ints'][0]
                fH.seek(0,2)
                fH.seek(fH.tell()-self.dtype.itemsize, 0)
                row = np.fromfile(fH, self.dtype)
                self.lastjd = row['ints'][0]
                self.jdrange =  self.lastjd - self.firstjd + 1
            except:
                self.firstjd = self.lastjd = self.jdrange = 0
                
    def load(self, ftype="run", stype='bin', part=None, filename='',
             jdstart=0, intstart=0, rawdata=False, nogmt=False,
             partappend=True):
        """Load a tracmass output file. Add data to class instance."""
        if jdstart != 0:
            ints = (jdstart+self.base_iso) * 24./self.nlgrid.ngcm  + 1
            filename = ("%s%08i_%s.%s" % (self.datafile,ints,ftype,stype))
            self.jd = jdstart
        elif intstart != 0:
            filename = ("%s%08i_%s.%s" % (self.datafile,intstart,
                                          ftype,stype))
        elif filename == '':
            filename = "%s_%s.%s" % (self.currfile[:-8],ftype,stype) 

        def readfile(filename):
            if filename[-3:] == "bin":
                return self.read_bin(self.datadir + filename)
            elif filename[-3:] == "asc":
                return self.read_asc(self.datadir + filename)
            else:
                raise IOError, "Unknown file format, file should be bin or asc"
                    
        partlist = self.list_partfiles(filename)
        if (len(partlist) > 1) & (partappend is True):
            runtraj = np.array([], dtype=self.dtype)
            for fname in partlist:
                runtraj = np.concatenate((runtraj, readfile(fname)))
        else:
            runtraj = readfile(filename)

        if rawdata is True: self.runtraj = runtraj
        self.filename = filename
        tvec = ['ntrac', 'ints', 'x', 'y', 'z']
        for tv in tvec:
            self.__dict__[tv] = runtraj[:][tv]
        #self.ints = self.ints.astype(np.int64)
        self.x = self.x - 1
        self.y = self.y - 1
        self.x[self.x<0] = self.x[self.x<0] + self.imt
        if hasattr(self.gcm,'gmt') & (nogmt is False):
            pos = self.imt - self.gcm.gmt.gmtpos
            x1 = self.x <= pos
            x2 = self.x >  pos
            self.x[x1] = self.x[x1] + self.gcm.gmt.gmtpos
            self.x[x2] = self.x[x2] - pos

        #mask = (self.x>self.imt+1) | (self.y>self.jmt+1)
        #for tv in tvec:
        #    self.__dict__[tv] = self.__dict__[tv][mask]
        self.intstart = intstart
        if self.nlrun.twritetype == 1:
            self.jd = (self.ints.astype(float)/60/60/24 + self.base_iso)
        elif self.nlrun.twritetype == 2:
            self.jd = (self.ints + self.base_iso)
        else:
            self.jd = (self.ints * self.nlgrid.ngcm/24. +self.base_iso) 
        self.jdvec = np.unique(self.jd)
        if hasattr(self,'lon'): self.ijll()

    @Traj.trajsloaded
    def fld2trajs(self, fieldname, mask=None, k=0):
        """Attach vaules from a model field to particles"""
        self.__dict__[fieldname] = self.x * 0
        for jd in np.unique(self.jd):
            jdmask = self.jd == jd
            tpos = self.gcm.load(fieldname, jd=jd+1)
            vec = self.gcm.ijinterp(self.x[jdmask], self.y[jdmask],
                                    self.gcm.__dict__[fieldname][k,:,:],nei=4)
            self.__dict__[fieldname][jdmask] = vec.astype(np.float32)
            jdmask = jdmask & (self.ntrac == 100)

    @Traj.trajsloaded
    def reldisp(self,mask=None):
        """Calculate the relative dispersion
           (mean square pair separation)
           Source: http://dx.doi.org/10.1357/002224009790741102

        """
        if mask is not None:
            mask = tr.x == tr.x
        xvec = self.x[mask]
        yvec = self.y[mask]
        jdvec = self.jd[mask]

        rsq = []
        for jd in np.unique(jdvec):
            jdmask = jdvec == jd
            mati,matj = np.meshgrid(xvec[jdmask], xvec[jdmask])
            xmat = (mati-matj)**2
            mati,matj = np.meshgrid(yvec[jdmask], yvec[jdmask])
            ymat = (mati-matj)**2
            rsq.append((ymat+xmat).sum(axis=None)/len(xvec*xvec-xvec))
        return np.array(rsq/rsq[0])
        
    def db_bulkinsert(self,datafile=None):
        """Insert trm bin-files data using pg_bulkload"""
        import batch
        pg_bulkload = "/opt/local/lib/postgresql90/bin/pg_bulkload"
        ctl_file = "load_trm.ctl"
        db = "-dpartsat"
        outtable = "-O" + "temp_bulkload" # self.tablename

        def run_command(datafile):
            t1 = dtm.now()
            sql = "truncate table temp_bulkload;"
            self.c.execute(sql)
            self.conn.commit()      

            infile = "-i%s/%s" % (self.datadir, datafile)
            spr.call([pg_bulkload,ctl_file,db,infile,outtable])
            print "Elapsed time: " + str(dtm.now()-t1)            

            sql = "SELECT min(ints),max(ints) FROM temp_bulkload;"
            self.c.execute(sql)
            jd1,jd2 = self.c.fetchall()[0]
            tablename = self.tablejds(jd1)
            runid = self.generate_runid(jd1=jd1, jd2=jd2,
                                        filename=datafile,
                                        tablename=tablename)
            print "Elapsed time: " + str(dtm.now()-t1)            

         
            self.db_create_partition(self.tablename, tablename)
            sql1 = "INSERT INTO %s (runid,ints,ntrac,x,y,z) " % tablename
            sql2 = "   SELECT %i as runid,ints,ntrac,x,y,z " % runid
            sql3 = "      FROM temp_bulkload;"
            self.c.execute(sql1 + sql2 + sql3)
            self.conn.commit()
            print "Elapsed time: " + str(dtm.now()-t1)

            batch.purge()
            
        if datafile:
            run_command(datafile)
        else:
            flist = glob.glob( "%s/%s*_run.bin" % (self.datadir,
                                                   self.datafile))
            for f in flist: run_command(os.path.basename(f))

    def create_seedfile(self,filename, k, mask):
        """Create a seed file based on a 2D mask for TRACMASS """
        ii,jj = np.meshgrid(np.arange(mask.shape[1]),
                            np.arange(mask.shape[0]))
        f = open(filename,'w')
        for i,j,m in zip(np.ravel(ii), np.ravel(jj), np.ravel(mask)):
            if m: f.writelines("% 6i% 6i% 6i% 6i% 6i% 6i\n" %
                               (i+1, j+1, k+1,3,0,50))
        f.close()

    @property
    def currfile(self, ftype='run', stype='bin'):
        flist = glob.glob("%s/%s*%s.%s" %
                          (self.datadir,self.nlrun.outDataFile,
                           ftype,stype))
        datearr = np.array([ os.path.getmtime(f) for f in flist])
        listpos = np.nonzero(datearr == datearr.max())[0][0]
        return os.path.basename(flist[listpos])

    def parse_filename(self,filename):        
        """Extract info about file from filename"""
        filename = os.path.basename(filename)
        fdict = {}
        arglist   = ['part','rank','arg1','arg2']
        for a in arglist: fdict[a] = -999
        plist = filename[len(self.nlrun.outDataFile)+1:].split('_')
        fdict['ftype'],fdict['stype'] = plist[-1].split('.')
        for n in plist[:-1]:
            if   'r' in n : fdict['rank'] = int(n[1:])
            elif 'p' in n : fdict['part'] = int(n[1:])
            elif 'a' in n : fdict['arg1'] = int(n[1:])
            elif 'b' in n : fdict['arg2'] = int(n[1:])
            elif 't' in n : fdict['ints'] = int(n[1:])
            else :
                try:
                    fdict['ints'] = int(n[1:])
                except:
                    fdict['stuff'] = n
        return fdict

    def list_partfiles(self, filename=None):
        """ Create list of all parts of a file"""
        if filename is None: 
            filename = self.currfile
        globname = '_'.join(['p*' if p[0] is 'p' else
                             p for p in filename.split('_') ])
        return [os.path.basename(g)
                for g in glob.glob(os.path.join(self.datadir,globname))]


    def gen_filelists(self, part=None, rank=None, arg1=None, arg2=None):
        """ Create list of all output files connected to current case"""
        arglist   = ['part','rank','arg1','arg2']
        ftypelist = ['run','err','in','out','kll']
        for tp in ftypelist:
            flist = glob.glob("%s/%s*_%s.*"%
                              (self.datadir, self.nlrun.outDataFile, tp))
            self.__dict__[tp + "files"] = flist
        self.filedict = {}
        tmpdict = {}
        for a in arglist: tmpdict[a] = []
        for f in self.runfiles:
            fd = self.parse_filename(f)
            for a in arglist:
                tmpdict[a].append(fd[a] if a in fd.keys() else -999)
            self.filedict[f] = fd
        if len(self.runfiles) > 0:
            for a in arglist: setattr(self, 'file%ss'%a, np.array(tmpdict[a]))
            mask = self.fileparts == self.fileparts
            for a,i in zip(arglist, [part, rank, arg1, arg2]):
                if i is not None:
                    mask = mask & (self.__dict__['file%ss'%a] == i)
            for a in arglist:
                setattr(self, 'file%ss'%a, self.__dict__['file%ss'%a][mask])
            for tp in ftypelist:
                flist =  self.__dict__[tp + "files"]
                if len(flist) == len(mask):
                    self.__dict__[tp + "files"] = list(np.array(flist)[mask])

    @property
    def part(self):
        return  self.__dict__['part']
    @part.setter
    def part(self, val):
        self.gen_filelists(part=val, rank=self.rank, arg1=self.arg1,
                           arg2=self.arg2)
        self.__dict__['part'] = val
    @property
    def partvec(self):
        if not hasattr(self, 'fileparts'): self.gen_filelists()
        return np.unique(self.fileparts)

    @property
    def rank(self):
        return  self.__dict__['rank']
    @rank.setter
    def rank(self, val):
        self.gen_filelists(part=self.part, rank=val, arg1=self.arg1,
                           arg2=self.arg2)
        self.__dict__['rank'] = val
    @property
    def rankvec(self):
        if not hasattr(self, 'fileranks'): self.gen_filelists()
        return np.unique(self.fileranks)

    @property
    def arg1(self):
        return  self.__dict__['arg1']
    @arg1.setter
    def arg1(self, val):
        self.gen_filelists(part=self.part, rank=self.rank, arg1=val,
                           arg2=self.arg2)
        self.__dict__['arg1'] = val
    @property
    def arg1vec(self):
        if not hasattr(self, 'filearg1s'): self.gen_filelists()
        return np.unique(self.filearg1s)

    @property
    def arg2(self):
        return  self.__dict__['arg2']
    @arg2.setter
    def arg2(self, val):
        self.gen_filelists(part=self.part, rank=self.rank, arg1=self.arg1,
                           arg2=val)
        self.__dict__['arg2'] = val
    @property
    def arg2vec(self):
        if not hasattr(self, 'filearg2s'): self.gen_filelists()
        return np.unique(self.filearg2s)
            
    @property
    def ls(self):
        self.gen_filelists()
        for f in self.runfiles: print f

    def __str__(self):
        """Print statistics about current instance"""
        alist = ['projname','casename', 'trmdir', 'datadir',
                 'njord_module', 'njord_class', 'imt', 'jmt']
        print ""
        print "="*79
        for a in alist:
            print a.rjust(15) + " : " + str(self.__dict__[a])
        if hasattr(self.nlrun, 'seedparts'):
            print "%s : %i" % ("seedparts".rjust(15),  self.nlrun.seedparts)
            print "%s : %i" % ("seedpart_id".rjust(15),self.nlrun.seedpart_id)

        for a in ['part','rank','arg1','arg2']:
            if self.__dict__[a] is not None:
                print "%s : %i" % (a.rjust(15),  self.__dict__[a])

        if hasattr(self,'x'):
            print ""
            print "%s : %s" % ("file".rjust(15), self.filename)
            print "%s : %s - %s" % (
                "time range".rjust(15),
                pl.num2date(self.jdvec.min()).strftime('%Y-%m-%d'),
                pl.num2date(self.jdvec.max()).strftime('%Y-%m-%d'))
            print "%s : %i" % ("timesteps".rjust(15), len(self.jdvec))
            print "%s : %i" % ("particles".rjust(15),
                               len(np.unique(self.ntrac)))
            print "%s : %i" % ("positions".rjust(15), len(self.ntrac))

        return ''
