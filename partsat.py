import os
import datetime
import glob
from datetime import datetime as dtm
from itertools import izip

import numpy as np
import pylab as pl
import scipy.io
from scipy.io import netcdf_file
from matplotlib.colors import LogNorm
import tables as td

from hitta import GBRY, WRY, GBW, GBW_R
import projmap, anim
import pytraj
import postgresql
import batch

miv = np.ma.masked_invalid

class Partsat(pytraj.Trm, postgresql.DB):

    def __init__(self, projname, casename="", **kwargs):
        super(Partsat,self).__init__(projname, casename, *kwargs)
        postgresql.DB.__init__(self, projname, casename, database='partsat')
        self.flddict = {'par':('L3',),'chl':('box8',)}
        if projname == 'oscar':
            import pysea.MODIS
            self.sat = pysea.NASA.nasa(res='4km',
                                       ijarea=(700,1700,2000,4000))
            def calc_jd(ints,intstart):
                return self.base_iso + float(ints)/6-1
        elif projname=="casco":
            self.sat = casco.Sat(res='500m')
            def calc_jd(ints,intstart):
                return (self.base_iso +(ints-(intstart)*10800)/150 +
                        intstart/8)
        elif projname=="gompom":
            n = netcdf_file('/Users/bror/svn/modtraj/box8_gompom.cdf')
            self.gomi = n.variables['igompom'][:]
            self.gomj = n.variables['jgompom'][:]
            self.sati = n.variables['ibox8'][:]
            self.satj = n.variables['jbox8'][:]
        elif projname=="jplSCB":
            from njord import mati
            self.sat = mati.Cal()
        elif projname=="jplNow":
            from njord import mati
            self.sat = mati.Cal()

    def map2grid(self, vec):
        """ Create map of average change in tracer """
        fld = self.grid(vec)
        cnt = self.grid()
        return fld/cnt

    def select(self,field="chl", jd=734107):
        """ Get the interpolated tracer for all trajectories at jd=jd"""
        dtype = np.dtype([('t1',np.float), ('t2',np.float),
                          ('val1',np.float), ('val2',np.float),
                          ('x',np.float),  ('y',np.float)])
        sql = """SELECT c.ints as t1,   c2.ints as t2,
                        c.val  as val1, c2.val  as val2,
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
        setattr(self, field, self.map2grid(getattr(self, field+'vec')))
        return True


    def sat_trajs(self,jd,field,pos='start'):
        """Retrive x-y pos and start-end values for trajs

        Functionality replaced by 'select

        '""" 
        if pos == 'start':
            t1str = " AND t.ints=t1.ints "; t2str = ""
        else:
            t2str = " AND t.ints=t2.ints "; t1str = ""
        table = self.tablename + '__' + field
        self.c.execute(
            "SELECT DISTINCT(ints) FROM %s WHERE runid=%i" %
            (table,intstart) )
        ints_t1t2 = self.c.fetchall()
        if len(ints_t1t2) < 2:
            self.empty = True
            return
        sql = """
        SELECT t.x as x , t.y as y, t.ntrac as n,
               t1.val as t1, t2.val as t2
           FROM gompom t
           INNER JOIN %s t1 ON
              t.intstart=t1.intstart %s AND t.ntrac=t1.ntrac
           INNER JOIN %s t2 ON
              t.intstart=t2.intstart %s AND t.ntrac=t2.ntrac
           WHERE t.intstart=%i AND t1.ints=%i AND t2.ints=%i
           """ % (table, t1str, table, t2str, intstart,
                  ints_t1t2[0][0], ints_t1t2[1][0])
        self.c.execute(sql)
        res = zip(*self.c.fetchall())
        if len(res) > 0:
            for n,a in enumerate(['x','y','ntrac','t1','t2']):
                self.__dict__[a] = np.array(res[n])
            self.empty = False
        else:
            self.empty = True
        self.ijll()

    def sat_conc(self,intstart,field,pos='start'):
        """Retrive fields of start- and end-values""" 
        if not hasattr(self, jdS):
            jdS = self.field_jds(field)
        if intstart in jdS.t0:
            if pos == 'start':
                ints = jdS.t1[jdS.t0==intstart].item()
            else:
                ints = jdS.t2[jdS.t0==intstart].item()
        else:            
            return self.llat * 0
        table = self.tablename + field
        sql = """
               SELECT round(t.x) x, round(t.y) y, avg(n.val) val
                  FROM gompom t
                  INNER JOIN %s n ON
                     t.intstart=n.intstart AND t.ntrac=n.ntrac
                  WHERE t.intstart=%i AND n.ints=%s AND t.ints=%s
                  GROUP BY round(t.x),round(t.y)
           """ % (table, intstart, intstart+1, ints)
        n = self.c.execute(sql)
        fld = self.llat * 0
        if n == 0: return fld
        x,y,val = zip(*self.c.fetchall())
        fld[np.array(y).astype(int),np.array(x).astype(int)-1] = val
        return fld

    def trajs(self,jdstart=0, jd=0, ntrac=0, fld=''):
        """ Retrive trajectories from database """
        whstr = ""
        if jd != 0:
            pass
            #whstr += " t.runid = %i AND" % runid
        if jd != 0:
            whstr += " t.ints = %i AND" % jd
        if ntrac != 0:
            whstr += " t.ntrac = %i " % ntrac
        whstr = whstr.rstrip("AND")
        if fld:
            valstr = " ,n.val val "
            table2 = self.tablename + "__" + fld
            valwhere = " AND n.ints=%s " % jd
            valjoin = (" INNER JOIN %s n ON " % table2 +
                       " t.runid=n.runid AND t.ntrac=n.ntrac ")
        else:
            valstr = " ,t.runid val "
            table2 = valwhere = valjoin = ''
        sql = ("""SELECT t.ints ints, t.ntrac ntrac, t.x x, t.y y
                 %s FROM %s t %s WHERE %s %s """
            % (valstr, self.tablename, valjoin, whstr, valwhere) )
        n = self.c.execute(sql)
        res = zip(*self.c.fetchall())
        if len(res) > 0:
               for n,a in enumerate(['ints','ntrac','x','y','val']):
                   self.__dict__[a] = np.array(res[n])
        #self.ijll()
  
    def get_satdata(self,field, jd=0):
        """ Load the satellite field corresponding to a given jd.
            Generate a vector with satellite data at all particle
            positions currently in memory """
        self.sat.load(field,jd=jd)
        self.ijll()
        self.sati,self.satj = self.sat.ll2ij(self.lon,self.lat)
        self.__dict__[field] = self.sat.__dict__[field][self.sati,self.satj]

    def create_fieldtable(self,field):
        """Create a postgresql table for satellite fields """
        tablename = self.tablename + '__' + field
        if self.table_exists(tablename): return
        sql = "CREATE TABLE %s (runid INT, ints FLOAT8, ntrac INT ,val REAL )"
        self.c.execute(sql % tablename)
        self.conn.commit()

    def insert_sat_to_db(self,field,jd1,jd2=None):
        """Insert field data into a table. """
        self.create_fieldtable(field)
        def insertload(jd):
            self.select(ints=jd)
            if not hasattr(self,'x'): return
            self.get_satdata(field,jd=jd)
            mask = ~np.isnan(self.__dict__[field])
            plist = zip(self.runid[mask], self.ints[mask], self.ntrac[mask],
                        self.__dict__[field][mask])
            tablename = self.tablename + '__' + field
            sql = ("INSERT INTO " + tablename + " (runid,ints,ntrac,val) " +
                   " VALUES (%s,%s,%s,%s)")
            self.c.executemany(sql,plist)
            self.conn.commit()
        if not jd2:
            insertload(jd1)
        else:
            for jd in np.arange(jd1,jd2+1):
                print jd2-jd
                insertload(jd)
                batch.purge()

    def field_jds(self,field):
        table = self.tablename + '__' + field
        class ints: pass
        self.c.execute("SELECT runid, min(ints), max(ints) " +
                       "FROM %s GROUP BY runid" % table)
        res = zip(*self.c.fetchall())
        if len(res) > 0:
            for n,a in enumerate(['t0','t1','t2']):
                ints.__dict__[a] = np.array(res[n])
        return ints

    def median_ncp(self):
        class svec: pass
        self.c.execute(
            """CREATE TEMPORARY TABLE IF NOT EXISTS temp_median 
                    (id INT AUTO_INCREMENT PRIMARY KEY) 
                SELECT ints, val FROM gompomncp 
                ORDER BY ints, val;
             """)
        self.c.execute(
            """CREATE TEMPORARY TABLE temp_median_ids 
               SELECT ROUND(AVG(id)) AS id FROM temp_median 
               GROUP BY ints;
            """)
        self.c.execute(
            """SELECT ints, val FROM temp_median_ids 
               LEFT JOIN temp_median USING (id) ORDER BY ints;
            """)
        res = zip(*self.c.fetchall())
        svec.t1 = res[0]
        svec.medianNCP = res[1]
        self.c.execute(
            """SELECT ints, count(val),max(val),min(val),avg(val) 
                   FROM gompomncp GROUP BY ints
                   ORDER BY ints;
            """)
        res = zip(*self.c.fetchall())
        svec.sum   = res[1]
        svec.max   = res[2]
        svec.min   = res[3]
        svec.mean  = res[4]
        
        return svec

    def calc__z_eu(self, chl):
        """ Calculate euphotic depth from chl with Morel's Case I model"""
        chl_tot = 40.2 * chl**0.507
        mask = chl < 1
	chl_tot[mask] = 38.0 * chl[mask]**0.425
        z_eu = 200.0 * chl_tot**-0.293
        mask = z_eu <= 102.0
        z_eu[mask] = 568.2 * chl_tot[mask]**-0.746
        return z_eu


class DeltaField(Partsat):
    """ Calculate change in tracer from one day to another """
    def __init__(self,projname, casename="", **kwargs):
        Partsat.__init__(self,projname,casename,**kwargs)
        self.jdmin = 733773.25
        self.jdmax = 734137.75
        self.djd = 1.
        self.h5dir = "./"
        
    def select(self,field="Dchl", jd=734107):
        """ Get the change in tracer for all trajectories at jd=jd"""
        if field[0] is not "D":
            return super(DeltaField,self).select(field, jd)
        field = field[1:]
        dtype = np.dtype([('dt',np.float),('val1',np.float),('val2',np.float),
                          ('x',np.float), ('y',np.float)])
        sql = """SELECT c2.ints-c.ints as dt, c.val as val1, c2.val as val2,
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

        field = 'D' + field 
        setattr(self, field + 'vec', (self.res['val2'] - self.res['val1']) /
                                self.res['dt'])
        self.x = self.res['x']
        self.y = self.res['y']
        setattr(self, field, self.map2grid(getattr(self, field + 'vec')))
        return True

    def get_ncp(self, jd=734107):
        """ get NCP from database"""
        if self.select('Dchl', jd=jd):
            z_eu = self.calc__z_eu(self.res['val1'])
            self.ncpvec = self.Dchlvec * z_eu * 71.3 / 12
            self.ncpvec[(self.ncpvec<-5000) | (self.ncpvec>5000)] = np.nan
            self.ncp = self.map2grid(self.ncpvec)

    def pcolor(self, field, jd=None, vmin=-10,vmax=10, oneside=False, clf=True, cb=True):
        """Plot a map of a field using projmap"""
        self.add_mp()
        if clf:
            pl.clf()
            pl.subplot(111,axisbg='0.9')
        if (oneside==True) | (oneside=="pos"):
            cmap = WRY()
        elif oneside=="neg":
            cmap = GBW()
        elif oneside=="negr":
            cmap = GBW_R()
        else:
            cmap = GBRY()
        self.mp.pcolormesh(self.mpxll,self.mpyll,miv(field), rasterized=True,
                           cmap=cmap, vmin=vmin,vmax=vmax)
        self.mp.nice(latlabels=False, lonlabels=False)
        if jd: pl.title(pl.num2date(jd).strftime("%Y-%m-%d"))
        if cb:
            pl.colorbar(aspect=40,shrink=0.95,pad=0,fraction=0.05)

    def movie(self, jd1, jd2, field='chl'):
        """Create a movie of the daily changes in tracer"""
        mv = anim.Movie()
        for jd in np.arange(jd1,jd2+1):
            t1 = dtm.now()
            print "Images left: ", jd2-jd
            self.map2grid(field=field, jd=jd)
            self.pcolor(self.dfld, jd)
            mv.image()
            print "Delta time: ", dtm.now() - t1
        mv.video(self.projname + "_" + field + "_mov.mp4",r=2)

    def h5array(self, field='Dchl'):
        """Create a tables array with gridded daily tracer changes"""
        self.h5open(field)
        for n,jd in enumerate(np.arange(self.jdmin, self.jdmax+1)):
            t1 = dtm.now()
            print "Images left: ", self.jdmax-jd
            if self.select(field=field, jd=jd):
                tpos = np.nonzero(self.h5f.root.jdvec==jd)[0][0]
                self.h5field[tpos,:,:] = getattr(self, field)
                print tpos
            print "Delta time: ", dtm.now() - t1
        self.h5close()
        
    def h5open(self, field):
        self.h5filename = os.path.join(self.h5dir,"partsat_%s_%s.h5" %
                                       (self.projname, self.casename))
        self.h5f = h5f = td.openFile(self.h5filename, 'a')
        jdvec = int((self.jdmax-self.jdmin+1)/self.djd)+1
        shape = (jdvec, self.jmt, self.imt)
        fatom = td.FloatCol()
        filtr = td.Filters(complevel=5, complib='zlib')
        crc = h5f.createCArray
        if not hasattr(h5f.root,  'jdvec'):
            jdvec = crc(h5f.root, 'jdvec',  fatom, (shape[0],))
            jdvec[:] = np.arange(self.jdmin, self.jdmax+1, self.djd)
        if not hasattr(h5f.root, field):
            fieldmat = crc(h5f.root, field, fatom,  shape, filters=filtr)
        self.h5field = self.h5f.root._f_getChild(field)

    def h5close(self):
        if not hasattr(self, 'h5f'): return
        self.h5f.close()
        del self.h5f
        del self.h5filename

    def histmoeller(self,fldname):
        """Create a hofmoeller representation of diff distributions"""
        sql = "SELECT min(ints),max(ints) FROM jplnowfull;"
        self.c.execute(sql)
        [jdmin,jdmax] = self.c.fetchall()[0]
        self.jdvec = np.arange(jdmin,jdmax)
        self.hpos = np.linspace(0, 100, 100)
        self.posmat = np.zeros((len(jdvec), len(self.hpos)-1))
        self.negmat = np.zeros((len(jdvec), len(self.hpos)-1))
        for n,jd in enumerate(self.jdvec):
            if not self.select(fldname,jd=jd): continue
            diff = (self.val2-self.val1)/self.dt
            self.negmat[n,:],_ = np.histogram(-diff[diff<0], self.hpos)
            self.posmat[n,:],_ = np.histogram(diff[diff>0], self.hpos)
            print n,jd,len(-diff[diff<0]),len(diff[diff>0])

    def cumcum_moeller(self,fldname):
        """Create a cumcum representation of diff distributions"""
        sql = "SELECT min(ints),max(ints) FROM jplnowfull;"
        self.c.execute(sql)
        [jdmin,jdmax] = self.c.fetchall()[0]
        jdvec = np.arange(jdmin,jdmax)
        self.poscum = np.zeros((len(jdvec), 100))
        self.negcum = np.zeros((len(jdvec), 100))
        self.pavglog =[]
        self.pavglin =[]
        self.navglog =[]
        self.navglin =[]
   
        for n,jd in enumerate(jdvec):
            if not self.select(fldname,jd=jd): continue
            _,self.negcum[n,:] = self.cumcum(-self.diff[self.diff<0])
            _,self.poscum[n,:] = self.cumcum(self.diff[self.diff>0])
            print n,jd,len(self.diff)
            self.pavglog.append(np.mean(np.log(self.diff[self.diff>0])))
            self.pavglin.append(np.mean(self.diff[self.diff>0]))
            self.navglog.append(np.mean(np.log(-self.diff[self.diff<0])))
            self.navglin.append(np.mean(-self.diff[self.diff<0]))

    def cumcum(self, vec):
        """Create a interpolated 'hypsograph' from a vector"""
        vec = vec[~np.isnan(vec)]
        xi = np.linspace(0,1,100)
        xvec = np.arange(len(vec)).astype(np.float)/len(vec)
        yvec = np.cumsum(np.sort(vec)[::-1].astype(np.float)/sum(vec))
        yi = np.interp(xi, xvec, yvec)
        return xi,yi

    def cumpos(self, vec, pos=0.1):
        """ Blaaaa"""
        vec = vec[~np.isnan(vec)]
        xvec = np.arange(len(vec)).astype(np.float)/len(vec)
        yvec = np.cumsum(np.sort(vec)[::-1].astype(np.float)/sum(vec))
        return np.interp(pos, xvec, yvec)


    def cumcumplot(self, fldname, jd=734107):
        """Plot a 'hypsograph of the relative distribution of diffs"""
        self.select(fldname, jd=jd)
        pl.clf()
        pl.plot(*self.cumcum(abs(np.random.rand(500000))))
        pl.plot(*self.cumcum(abs(np.random.randn(500000))))
        pl.plot(*self.cumcum(-self.diff[self.diff<0]))
        pl.plot(*self.cumcum( self.diff[self.diff>0])) 
        pl.xlim(0,1)
        pl.ylim(0,1)
        pl.title("%s %s" % (fldname, pl.num2date(jd).strftime('%Y-%m-%d')))
        pl.legend(('Random values, symetrical', 'Random values, gaussian',
                   'Trajs, increasing values','Trajs, decreasing values'))
        pl.savefig('figs/cumcumplot_%s_%i.pdf' % (fldname, jd))















#####################################################################
def ncp(t, db=False,kpar=False):
        class tr: pass
        t = int(t)
        tr_chl = traj('gompom')
        tr_chl.sat_trajs(t, 'chlor_a')
        tr_chl2 = traj('gompom') 
        tr_chl2.sat_trajs(t, 'chlor_a', pos='end')
        if tr_chl.empty: return False
        tr_k49 = traj('gompom')
        tr_k49.sat_trajs(t, 'K_490')
        if kpar:
            """
http://oceancolor.gsfc.nasa.gov/forum/oceancolor/topic_show.pl?tid=2997
            """
            tr_k49.t1 = 0.0864 + 0.884*tr_k49.t1 - 0.00137/tr_k49.t1
            tr_k49.t2 = 0.0864 + 0.884*tr_k49.t2 - 0.00137/tr_k49.t2
        msk = np.intersect1d(tr_chl.ntrac,tr_k49.ntrac)
        tr_k49.msk = [np.flatnonzero(tr_k49.ntrac==m).item()
                      for m in msk]
        tr_chl.msk = [np.flatnonzero(tr_chl.ntrac==m).item()
                      for m in msk]
        tr_k49.eu1 = float(4.6) / tr_k49.t1
        tr_k49.eu2 = float(4.6) / tr_k49.t2
        if len(tr_chl.msk) == 0: return False
        jds =tr_chl.field_jds('chlor_a')

        tr.pc2 = tr_chl.t2[tr_chl.msk] * tr_k49.eu2[tr_k49.msk]*60.
        tr.pc1 = tr_chl.t1[tr_chl.msk] * tr_k49.eu1[tr_k49.msk]*60.
        dt = (jds.t2[jds.t0==t]/6. - jds.t1[jds.t0==t]/6.).item()

        tr.ncp = (tr.pc2-tr.pc1) / dt
        tr.ntrac = tr_chl.ntrac[tr_chl.msk]
        tr.x1 = tr_chl.x[tr_chl.msk]
        tr.y1 = tr_chl.y[tr_chl.msk]
        tr.x2 = tr_chl2.x[tr_chl.msk]
        tr.y2 = tr_chl2.y[tr_chl.msk]
        tr.lon1 = tr_chl.lon[tr_chl.msk]
        tr.lat1 = tr_chl.lat[tr_chl.msk]
        tr.lon2 = tr_chl2.lon[tr_chl.msk]
        tr.lat2 = tr_chl2.lat[tr_chl.msk]

        if db:
            tr.ints=tr.x1*0+t
            tr.val = tr.ncp
            tr_chl.sat_to_db('ncp',intstart=t,ints=t,batch=False,traj=tr)
        return tr

def batch_sat_to_db(field='chlor_a',batchprefix='batch_ints'):
    tr = traj('gompom')
    file1 = tr.ormdir + "/projects/gomoos/" + batchprefix + "_start.asc"
    file2 = tr.ormdir + "/projects/gomoos/" + batchprefix + "_end.asc"
    for t1,t2 in zip(open(file1),open(file2)):
        print int(t1), int(t1)+1, int(t1) ,int(t2), int(t2)-int(t1)
        if int(t1) > 4000:
            try:
                tr.sat_to_db(field,int(t1),int(t1)+1,batch=True)
            except IOError:
                print "*** File missing! ***"
            try:
                tr.sat_to_db(field,int(t1),int(t2),batch=True)
            except IOError:
                print "*** File missing! ***"
    tr.enable_indexes()

def calc_ncp_time(batchfile='batch_ints_start.asc'): 
    import partsat
    class tS: pass
    tr = partsat.traj('gompom')
    tS.ncp = []
    tS.iso = []
    tS.cnt = []
    for t in open(tr.ormdir + "/projects/gomoos/" + batchfile):
        ncp = tr.traj_ncp(int(t))
        if not ncp: continue
        tS.ncp.append(np.median(ncp.ncp))
        tS.iso.append(tr.ints2iso(int(t)))
        tS.cnt.append(len(ncp.ncp))
    return tS

def interp(ps,intstart=6353):
    import figpref
    import projmaps
    figpref.current()
    
    ps.c.execute("SELECT distinct(ints) FROM casco__chlor_a " + 
                 "   WHERE intstart=%i" % intstart) 
    ints1,ints2 = ps.c.fetchall()

    sql = """SELECT t1.ntrac,t1.val,t2.val,p.x,p.y,p.ints
               FROM casco__chlor_a t1
               INNER JOIN casco__chlor_a t2 ON t1.ntrac=t2.ntrac
               INNER JOIN casco p ON t1.ntrac=p.ntrac
               WHERE t1.intstart=%i
                  AND t2.intstart=%i
                  AND p.intstart=%i
                  AND t1.ints=%i
                  AND t2.ints=%i;
                  """ % (intstart,intstart,intstart,ints1[0],ints2[0])
    ps.c.execute(sql)
    res = zip(*ps.c.fetchall())
    class trj: pass
    if len(res) > 0:
        for n,a in enumerate(['ntrac','t1val','t2val','x','y','ints']):
            trj.__dict__[a] = np.array(res[n])

    mask = (trj.ints.astype(np.float)/100-trj.ints/100)>0.5
    trj.ints[mask]=trj.ints[mask]+1
    tvec = np.unique(trj.ints)
    tvec = tvec[tvec<=ints2[0]]
    itvec = tvec-tvec.min()
    itvec = itvec.astype(np.float)/itvec.max()

    mp = projmaps.Projmap('casco')
    xl,yl = mp(ps.llon,ps.llat)
    for n,t in enumerate(tvec):
        fld = ps.cs.llat * 0
        cnt = ps.cs.llat * 0

        xvc = (trj.x[trj.ints==t].astype(np.int))
        yvc = (trj.y[trj.ints==t].astype(np.int))
        val = ( np.log(trj.t1val[trj.ints==t])*(1-itvec[n]) +
                np.log(trj.t2val[trj.ints==t])*(itvec[n]) )
        for x,y,v in zip(xvc,yvc,val):
            fld[y,x] += v
            cnt[y,x] += 1

        pl.clf()
        mp.pcolormesh(xl,yl,miv(fld/cnt))
        mp.nice()
        jd = (ps.base_iso +
              (float(t)-(intstart)/8*3600*24)/3600/24 + intstart/8) + 0.583
        pl.title(pl.num2date(jd).strftime("log(Chl) %Y-%m-%d %H:%M"))
        pl.clim(-2,2.5)
        pl.savefig("interp_%i_%03i.png" % (intstart,n), dpi=100)


def batch_insert():
    import batch
    def copy(jd):
        tr = traj('jplNOW','ftp','/Volumes/keronHD3/ormOut/')
        print pl.num2date(jd), jd
        tr.load(jd)
        tr.remove_satnans()
        if len(tr.x>0):
            tr.db_copy()

    #batch.jdloop(copy,733773.0, 734138.0,3)
    for jd in np.arange(733865.0,734138):
        dt1 = pl.date2num(dtm.now())
        copy(jd)
        dt2 = pl.date2num(dtm.now())        
        print "----------",dt2-dt1

def profile():
    import cProfile
    cProfile.run('test()')
