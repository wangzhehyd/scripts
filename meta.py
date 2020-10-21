"""
Module for all metadynamics backend functionality

"""

# Copyright Schrodinger, LLC. All rights reserved.

#- Imports -------------------------------------------------------------------

import math
import sys
from past.utils import old_div

import numpy

from schrodinger.application.desmond.cms import Cms
from schrodinger.application.desmond.constants import SIM_BOX
from schrodinger.application.desmond.enhsamp import parseStr
from schrodinger.infra import mmjob
from schrodinger.job import jobcontrol
from schrodinger.utils import sea

#- Globals -------------------------------------------------------------------

_backend_singleton = None

#- Functions -----------------------------------------------------------------


def get_backend():
    """
    A convenience function to see if we're running under job control. If so,
    return a _Backend object. Otherwise, return None.

    """
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton

    backend = _Backend()
    if backend.job_id:
        _backend_singleton = backend
        return backend
    return None


class _Backend(jobcontrol._Backend):

    def __init__(self):

        super(_Backend, self).__init__()

    def setJobProgress(self, steps=0, totalsteps=0, description=""):
        """
        Tell jmonitor to set the job progress (steps out of total steps).
        Set either steps and totalsteps, or description, or both.
        """
        mmjob.mmjobbe_set_jobprogress(steps, totalsteps, description)


class CV:
    """
    base class for collective variable
    """

    def __init__(self, dim, width, wall, floor):
        self.dim = dim
        self.width = width
        self.wall = wall
        self.floor = floor

    def getMExpr(self, model, cvid):
        raise NotImplementedError("")


class CVrgyr(CV):
    """
    Radius of Gyration Collective Variable
    """
    cvrgyr_template = \
"""
#radius of gyration definition
%(cvname)s_sel = %(atomlist)s;
%(cvname)s_cog = center_of_geometry(%(cvname)s_sel);
%(cvname)s_coord_range = series (i=0:length(%(cvname)s_sel))
        norm2(min_image(pos(%(cvname)s_sel[i])-%(cvname)s_cog));
%(cvname)s=sqrt(%(cvname)s_coord_range/length(%(cvname)s_sel));
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self, atomlist, width):
        CV.__init__(self, 1, width, None, None)
        self._atomlist = atomlist

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        return self.cvrgyr_template % template_dict


class CVrgyr_mass(CV):
    """
    Radius of Gyration Collective Variable
    """
    cvrgyr_mass_template = \
"""
#mass-weighted radius of gyration definition
%(cvname)s_sel = %(atomlist)s;
%(cvname)s_com = center_of_mass(%(cvname)s_sel);
%(cvname)s_coord_range = series (i=0:length(%(cvname)s_sel))
        mass(%(cvname)s_sel[i])*norm2(min_image(pos(%(cvname)s_sel[i])-%(cvname)s_com));
%(cvname)s=sqrt(%(cvname)s_coord_range/sum(mass(%(cvname)s_sel)));
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s

"""

    def __init__(self, atomlist, width):
        CV.__init__(self, 1, width, None, None)
        self._atomlist = atomlist

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        return self.cvrgyr_mass_template % template_dict


class CVrmsd(CV):
    """
    rmsd collective variable
    """
    cvrmsd_template = \
"""
# rmsd definition
%(cvname)s_sel = %(atomlist)s;
%(cvname)s_ref = array( %(xyz_ref)s );
%(cvname)s = rmsd( %(cvname)s_ref,  %(cvname)s_sel );
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""
    cvrmsd_weights_template = \
"""
# rmsd definition
%(cvname)s_sel = %(atomlist)s;
%(cvname)s_ref = array( %(xyz_ref)s );
%(cvname)s_rmsd_weights = array( %(rmsd_weights)s );
%(cvname)s_superpos_weights = array( %(superpos_weights)s );
%(cvname)s = rmsd( %(cvname)s_ref,  %(cvname)s_sel, %(cvname)s_rmsd_weights, %(cvname)s_superpos_weights );
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self,
                 atomlist,
                 xyz_coords_ref,
                 width,
                 rmsd_weights=None,
                 superpos_weights=None):
        CV.__init__(self, 1, width, None, None)
        self._atomlist = atomlist
        self._xyz_coords_ref = xyz_coords_ref
        self._rmsd_weights = rmsd_weights
        self._superpos_weights = superpos_weights

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['xyz_ref'] = list2str(self._xyz_coords_ref)
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        if (self._rmsd_weights is not None and
                self._superpos_weights is not None):
            template_dict['rmsd_weights'] = list2str(self._rmsd_weights)
            template_dict['superpos_weights'] = list2str(self._superpos_weights)
            return self.cvrmsd_weights_template % template_dict
        elif (self._rmsd_weights is not None or
              self._superpos_weights is not None):
            raise RuntimeError("Superposition and RMSD weights must both be " +
                               "defined if using either one")
        return self.cvrmsd_template % template_dict


class CVrmsd_symm(CV):
    """
    rmsd collective variable
    """
    cvrmsd_setup_template = \
"""
# setting up rmsd_symm with total of %(nconfs)s conformations
%(cvname)s_sel = %(atomlist)s;

"""
    cvrmsd_conf_template = \
"""
# rmsd_symm definition #%(confnum)s
%(cvname)s_ref_%(confnum)s = array( %(xyz_ref)s );
%(cvname)s_%(confnum)s     = rmsd( %(cvname)s_ref_%(confnum)s, %(cvname)s_sel );
"""
    cvrmsd_template = \
"""
%(cvname)s = min( array( %(confs)s ));
print ("%(cvname)s", %(cvname)s );
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self, atomlist, xyz_coords_ref_list, width):
        CV.__init__(self, 1, width, None, None)
        # atomlist will b a list of atomlist
        self._atomlist = atomlist
        self._xyz_coords_ref = xyz_coords_ref_list

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        nconfs = len(self._xyz_coords_ref)
        template_dict['nconfs'] = nconfs
        mexpression = self.cvrmsd_setup_template % template_dict
        for conf_num in range(nconfs):
            template_dict['xyz_ref'] = list2str(self._xyz_coords_ref[conf_num])
            template_dict['confnum'] = conf_num
            mexpression += self.cvrmsd_conf_template % template_dict
        confs_str = ''
        for i in range(nconfs):
            confs_str += cvname + '_' + str(i) + ', '
        template_dict['confs'] = confs_str[0:len(confs_str) - 2]
        mexpression += self.cvrmsd_template % template_dict
        return mexpression


class CVwhim(CV):
    """
    whim collective variable
    """
    cvwhim_template = \
"""
#whim definition
%(cvname)s_sel  = %(atomlist)s;
%(cvname)s_whim = whim(%(cvname)s_sel, mass(%(cvname)s_sel));
%(cvname)s = %(cvname)s_whim[%(eigval)i];
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self, atomlist, eigval, width):
        CV.__init__(self, 1, width, None, None)
        self._atomlist = atomlist
        self._eigval = eigval - 1

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['eigval'] = self._eigval
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        return self.cvwhim_template % template_dict


class CVzdist0(CV):
    """
    This collective variable reports an absolute Z-distance from the simulation box
    origin (Z==0).  This cv is useful when for membrane penetration studies.
    """
    cvzdist0_template = \
"""
# Z-dist definition
%(cvname)s_g0 = center_of_mass ( %(atomlist)s );
%(cvname)s_z  = %(cvname)s_g0[2];
%(cvname)s    = sqrt(%(cvname)s_z^2);
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self, atomlist, width):
        CV.__init__(self, 1, width, None, None)
        self._atomlist = atomlist

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        return self.cvzdist0_template % template_dict


class CVzdist(CV):
    """
    This collective variable reports an absolute Z-distance.
    this CV is used for membrane penetration studies.
    """
    cvzdist_template = \
"""
# Z-dist definition
%(cvname)s_g0 = center_of_mass ( %(atomlist)s );
%(cvname)s  = %(cvname)s_g0[2];
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""
    #in gnuplot: 'plot [0:10] 1000/(1+exp((4-x)/ 0.2))'
    cvdist_wall_template = \
"""
# the upper bound, wall params for %(cvname)s are: width is 0.2;
# location at %(wall)f; hight is 1000
%(cvname)s_wall = 1000 / (1 + exp(( %(wall)f - %(cvname)s) / 0.2) );
"""

    #in gnuplot: 'plot [0:10] 1000/(1+exp((x-4)/ 0.2))'
    cvdist_floor_template = \
"""
# lower bound wall or 'floor' params for %(cvname)s are: width is 0.2;
# location at %(floor)f; # hight is 1000
%(cvname)s_floor = 1000 / (1 + exp((%(cvname)s - %(floor)f) / 0.2) );
"""

    def __init__(self, atomlist, width, wall, floor):
        CV.__init__(self, 1, width, wall, floor)
        self._atomlist = atomlist

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['atomlist'] = model.atid2atomsel(self._atomlist)
        mexpr_wall = ''
        mexpr_floor = ''
        if (self.wall is not None):
            template_dict['wall'] = self.wall
            mexpr_wall = self.cvdist_wall_template % template_dict
        if (self.floor is not None):
            template_dict['floor'] = self.floor
            mexpr_floor = self.cvdist_floor_template % template_dict

        return self.cvzdist_template % template_dict + mexpr_wall + mexpr_floor


class CVDist(CV):
    """
    distance collective variable
    """
    cvdist_template = \
"""
# distance definition
%(cvname)s_p0 = %(p0_atomsel)s;
%(cvname)s_p1 = %(p1_atomsel)s;
%(cvname)s = dist(%(cvname)s_p0, %(cvname)s_p1);
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""

    cvdist_grp_template = \
"""
# distance definition for group of atoms
%(cvname)s_g0 = center_of_mass ( %(p0_atomsel)s );
%(cvname)s_g1 = center_of_mass ( %(p1_atomsel)s );
%(cvname)s = norm(min_image(%(cvname)s_g0 - %(cvname)s_g1));
print ("%(cvname)s", %(cvname)s);
# the width for %(cvname)s will be set to: %(width)s
"""
    #in gnuplot: 'plot [0:10] 1000/(1+exp((4-x)/ 0.2))'
    cvdist_wall_template = \
"""
# the upper bound, wall params for %(cvname)s are: width is 0.2;
# location at %(wall)f; hight is 1000
%(cvname)s_wall = 1000 / (1 + exp(( %(wall)f - %(cvname)s) / 0.2) );
"""

    #in gnuplot: 'plot [0:10] 1000/(1+exp((x-4)/ 0.2))'
    cvdist_floor_template = \
"""
# lower bound wall or 'floor' params for %(cvname)s are: width is 0.2;
# location at %(floor)f; # hight is 1000
%(cvname)s_floor = 1000 / (1 + exp((%(cvname)s - %(floor)f) / 0.2) );

"""

    def __init__(self, p0, p1, width, wall, floor):
        CV.__init__(self, 1, width, wall, floor)
        self._p0 = p0
        self._p1 = p1
        self._groups = False
        if ((not (isinstance(p0, int))) | (not (isinstance(p1, int)))):
            self._groups = True

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['p0_atomsel'] = model.atid2atomsel(self._p0)
        template_dict['p1_atomsel'] = model.atid2atomsel(self._p1)
        mexpr_wall = ''
        mexpr_floor = ''
        if (self._groups):
            mexpr_dist = self.cvdist_grp_template % template_dict
        else:
            mexpr_dist = self.cvdist_template % template_dict

        if (self.wall is not None):
            template_dict['wall'] = self.wall
            mexpr_wall = self.cvdist_wall_template % template_dict
        if (self.floor is not None):
            template_dict['floor'] = self.floor
            mexpr_floor = self.cvdist_floor_template % template_dict
        return mexpr_dist + mexpr_wall + mexpr_floor


class CVAngle(CV):
    """
    A class to define angle collective variable.
    Note that due to numerical instability, cosine of the angle is used
    instead of radian.
    """
    cvangle_template = \
"""
# angle definition
%(cvname)s_p0 = %(p0_atomsel)s;
%(cvname)s_p1 = %(p1_atomsel)s;
%(cvname)s_p2 = %(p2_atomsel)s;
%(cvname)s = angle_gid(%(cvname)s_p0, %(cvname)s_p1, %(cvname)s_p2);
print ("%(cvname)s", acos(%(cvname)s) );
# the width for %(cvname)s will be set to: %(width)s
"""
    cvangle_grp_template = \
    """
# angle definition for group of atoms
%(cvname)s_g0 = center_of_mass ( %(p0_atomsel)s );
%(cvname)s_g1 = center_of_mass ( %(p1_atomsel)s );
%(cvname)s_g2 = center_of_mass ( %(p2_atomsel)s );
%(cvname)s_v0 = min_image(%(cvname)s_g0 - %(cvname)s_g1);
%(cvname)s_v1 = min_image(%(cvname)s_g2 - %(cvname)s_g1);
%(cvname)s = angle(%(cvname)s_v0, %(cvname)s_v1);
print ("%(cvname)s", acos(%(cvname)s) );
# the width for %(cvname)s will be set to: %(width)s
    """

    #in gnuplot: 'plot [0:10] 1000/(1+exp((2.8-x)/0.05))'
    cvangle_wall_template = \
"""
# the upper bound, wall params for %(cvname)s are: width is 0.57 degree;
# location at %(wall)f; hight is 1000
%(cvname)s_wall = 1000 / (1 + exp(( %(wall)f - acos(%(cvname)s))/0.05) );
"""

    #in gnuplot: 'plot [0:10] 1000/(1+exp((x-1.4)/0.05))'
    cvangle_floor_template = \
"""
# lower bound wall or 'floor' params for %(cvname)s are: width is 0.57 degree;
# location at %(floor)f; # hight is 1000
%(cvname)s_floor = 1000/(1 + (exp((acos(%(cvname)s)- %(floor)f)/0.05)) );
"""

    def __init__(self, p0, p1, p2, width, wall, floor):
        CV.__init__(self, 1, width, wall, floor)
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        self._groups = False
        if ((not (isinstance(p0, int))) | (not (isinstance(p1, int))) |
            (not (isinstance(p2, int)))):
            self._groups = True

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['p0_atomsel'] = model.atid2atomsel(self._p0)
        template_dict['p1_atomsel'] = model.atid2atomsel(self._p1)
        template_dict['p2_atomsel'] = model.atid2atomsel(self._p2)
        mexpr_wall = ''
        mexpr_floor = ''
        if (self._groups):
            mexpr_angle = self.cvangle_grp_template % template_dict
        else:
            mexpr_angle = self.cvangle_template % template_dict

        if (self.wall is not None):
            template_dict['wall'] = self.wall
            mexpr_wall = self.cvangle_wall_template % template_dict
        if (self.floor is not None):
            template_dict['floor'] = self.floor
            mexpr_floor = self.cvangle_floor_template % template_dict
        return mexpr_angle + mexpr_wall + mexpr_floor


class CVDihedral(CV):
    """
    A class to define dihedral collective variable.
    Note that this collective variable is a two dimensional one.
    The first element is the cosine of the dihedral,
    and the second element is the sine of the dihedral angle.
    """
    cvdihedral_template = \
"""
# dihedral definition
%(cvname)s_p0 = %(p0_atomsel)s;
%(cvname)s_p1 = %(p1_atomsel)s;
%(cvname)s_p2 = %(p2_atomsel)s;
%(cvname)s_p3 = %(p3_atomsel)s;
%(cvname)s = dihedral_gid(%(cvname)s_p0, %(cvname)s_p1, %(cvname)s_p2, %(cvname)s_p3);
print ("%(cvname)s", atan2(%(cvname)s));
# the width for %(cvname)s will be set to: %(width)s
"""

    cvdihedral_grp_template = \
"""
# dihedral definition for group of atoms
%(cvname)s_g0 = center_of_mass ( %(p0_atomsel)s);
%(cvname)s_g1 = center_of_mass ( %(p1_atomsel)s);
%(cvname)s_g2 = center_of_mass ( %(p2_atomsel)s);
%(cvname)s_g3 = center_of_mass ( %(p3_atomsel)s);

%(cvname)s_v0 = min_image(%(cvname)s_g1 - %(cvname)s_g0);
%(cvname)s_v1 = min_image(%(cvname)s_g2 - %(cvname)s_g1);
%(cvname)s_v2 = min_image(%(cvname)s_g3 - %(cvname)s_g2);

%(cvname)s = dihedral(%(cvname)s_v0, %(cvname)s_v1, %(cvname)s_v2);
print ("%(cvname)s", atan2(%(cvname)s[1],%(cvname)s[0]));
# the width for %(cvname)s will be set to: %(width)s
"""

    def __init__(self, p0, p1, p2, p3, width, wall, floor):
        CV.__init__(self, 2, width, wall, floor)
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        self._groups = False
        if ((not (isinstance(p0, int))) | (not (isinstance(p1, int))) |
            (not (isinstance(p2, int))) | (not (isinstance(p3, int)))):
            self._groups = True

    def getMExpr(self, model, cvname):
        template_dict = {}
        template_dict['cvname'] = cvname
        template_dict['width'] = self.width
        template_dict['p0_atomsel'] = model.atid2atomsel(self._p0)
        template_dict['p1_atomsel'] = model.atid2atomsel(self._p1)
        template_dict['p2_atomsel'] = model.atid2atomsel(self._p2)
        template_dict['p3_atomsel'] = model.atid2atomsel(self._p3)
        if (self._groups):
            mexpr_dihed = self.cvdihedral_grp_template % template_dict
        else:
            mexpr_dihed = self.cvdihedral_template % template_dict
        return mexpr_dihed


class CmsModel:

    def __init__(self, model):
        self._model = model

    def atid2atomsel(self, atid):
        if isinstance(atid, int):
            return 'atomsel("atom. %d")' % atid
        return 'atomsel("atom. %s")' % list2str(atid)


def list2str(l):
    s = ''
    n = len(l)
    if n > 0:
        for e in l[:-1]:
            s += str(e) + ', '
        s += str(l[-1])
    return s


class Meta:
    declare_template =\
"""
declare_meta(
    dimension = %(dimension)d,
    cutoff    = %(cutoff)f,
    first     = %(first)f,
    interval  = %(interval)f,
    name      = "%(meta_name)s",
    initial   = "");

declare_output(
    name = "%(output_name)s",
    first = %(first)f,
    interval= %(interval)f);
"""

    meta_template =\
"""
# height used for this run is: %(height)f
meta(0, %(height_width)s,
        %(cv)s);
"""

    meta_well_tempered_template = \
"""
# height used for this run is: %(height)f, sampling temperature kT is: %(kTemp)f.
meta(0,
     array( %(height)f * exp( meta(0, %(height_width)s, %(cv)s )/(-1.0 * %(kTemp)f) ), %(width)s ),
     %(cv)s);
"""

    def __init__(self):
        self._cvs = []
        self.height = 0.0
        self.first = 0.0
        self.interval = 0.04
        self.cutoff = 9.0
        self.kTemp = -1.0
        self.name = "kerseq"
        self.cv_name = "cvseq"

    def generateCfg(self, model=None):
        mexpr = self._getMExpr(CmsModel(model))
        #print mexpr
        cfg_str = parseStr(model, mexpr)
        #print cfg_str
        #sys.exit(1)
        return cfg_str

    def _getMExpr(self, model):
        dim = sum([cv.dim for cv in self._cvs])
        width = []
        for cv in self._cvs:
            w = cv.width
            if isinstance(w, int) or isinstance(w, float):
                width.append(w)
            elif isinstance(w, list):
                width.extend(w)
            else:
                raise TypeError

        height = self.height
        kTemp = self.kTemp

        template_dict = {}
        template_dict['first'] = self.first
        template_dict['interval'] = self.interval
        template_dict['cutoff'] = self.cutoff
        template_dict['meta_name'] = self.name
        template_dict['output_name'] = self.cv_name
        template_dict['height'] = height
        template_dict['dimension'] = dim

        if (kTemp < 0):
            height_width = [height]
            height_width.extend(width)
            template_dict['height_width'] = 'array(%s)' % list2str(height_width)
        else:
            template_dict['width'] = list2str(width)
            template_dict['kTemp'] = kTemp
            height_width = [0.0]
            for i in range(len(width)):
                height_width.append(0.0)
            template_dict['height_width'] = 'array(%s)' % list2str(height_width)

        # declare statement must show up before anything else
        mexpr = [self.declare_template % template_dict]

        # define collective variables
        hw_varname = ['height']
        cv_varname = []
        for i, cv in enumerate(self._cvs):
            name = "cv_%02d" % i
            mexpr.append(cv.getMExpr(model, name))
            cv_varname.append(name)
            hw_varname.append(name + '_width')

        template_dict['cv'] = 'array(%s)' % list2str(cv_varname)
        for i, cv in enumerate(self._cvs):
            if (cv.wall is not None):
                name = "cv_%02d_wall" % i
                mexpr.append(name + " +  # add a wall")
            if (cv.floor is not None):
                name = "cv_%02d_floor" % i
                mexpr.append(name + " +  # add a floor")

        if (kTemp < 0):  # if well-tempered kTemp value is defined
            mexpr.append(self.meta_template % template_dict)
        else:
            mexpr.append(self.meta_well_tempered_template % template_dict)
        return '\n'.join(mexpr)

    def addCV(self, cv):
        self._cvs.append(cv)


meta_def_v2_sample = \
"""
meta = {
  interval = 0.04
  height = 0.5
  kTemp = 2.4
  name = "kerseq"
  cv_name = "cvseq"
  cv = [
    {
      type = dist
      atom = [1 3]
      width = 0.4
      ktemp = 2.4  # well-tempering option
      wall = 10.0  # upper bound for the dist CV
      floor =  3.0 # lower bound for the dist CV
    }
    {
      type = angle
      atom = [1 3 5]
      width = 0.4
    }
    {
      type = dihedral
      atom = [1 3 5 7]
      width = 0.4
    }
    {
      type = rmsd
      atom = [1 3 4 5 5 6 8 10]
      width = 0.4
    }
    {
      type = rmsd_alt
      atom = [1 3 4 5 5 6 8 10]
      width = 0.4
    }
    {
      type = rmsd_symm
      atom = [1 2 3 4 5 6 7 8 9 10]
      width = 0.4
    }
    { type = zdist0
      atom = [1 2 3 4 5]
      width = 0.5
    }
    { type = zdist
      atom = [1 2 3 4 5]
      width = 0.5
    }
  ]
}

"""


def parse_meta(m, model):

    def get_atom_list(cv_atom):
        atom_list = []
        for atom in cv_atom:
            atom_list.append(atom.val)
        return atom_list

    def set_parameter(meta, meta_sea, key):
        if key in meta_sea:
            setattr(meta, key, meta_sea[key].val)

    meta = Meta()
    set_parameter(meta, m, 'interval')
    set_parameter(meta, m, 'first')
    set_parameter(meta, m, 'height')
    set_parameter(meta, m, 'cv_name')
    set_parameter(meta, m, 'name')
    set_parameter(meta, m, 'cutoff')
    set_parameter(meta, m, 'kTemp')

    if model is None:
        return meta

    for cv in m.cv:
        cv_type = cv['type'].val.lower()
        wall = None
        floor = None
        if cv_type == 'dist':
            atom_list = get_atom_list(cv.atom)
            if len(atom_list) != 2:
                raise RuntimeError(
                    "Metadynamics: Distance-CV requires exactly 2 atoms/sites. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            if 'wall' in cv:
                wall = cv['wall'].val
            if 'floor' in cv:
                floor = cv['floor'].val
            # Check if the distances between the atoms/sites is:
            #            floor <= dist(a,b) <= wall
            # TODO: Periodicity of the system is not taken into the
            # account. center_of_mass function needs to to be expanded to do
            # this (Ev:125371).
            currentVal = get_distance(model, atom_list)
            if wall is not None and wall > 0 and wall <= currentVal:
                raise RuntimeError(
                    "Metadynamics: the current Distance-CV value (" + str(
                        currentVal) +
                    " Ang) exceeds the specified \'wall\' value (" + str(wall) +
                    " Ang).  Please increase the wall distance and resubmit your job."
                )
            if floor is not None and floor > 0 and floor >= currentVal:
                raise RuntimeError(
                    "Metadynamics: the current Distance-CV value (" + str(
                        currentVal) +
                    " Ang) is less than the specified \'floor\' value (" + str(
                        wall) +
                    " Ang).  Please reduce the floor distance and resubmit your job."
                )
            meta.addCV(CVDist(atom_list[0], atom_list[1], width, wall, floor))

        elif cv_type == "angle":
            atom_list = get_atom_list(cv.atom)
            if len(atom_list) != 3:
                raise RuntimeError(
                    "Metadynamics: Angle-CV requires exactly 3 atoms/sites. You selected: "
                    + str(len(atom_list)))
            width = (cv['width'].val) * math.pi / 180.00
            if 'wall' in cv:
                wall = cv['wall'].val * math.pi / 180.00
            if 'floor' in cv:
                floor = cv['floor'].val * math.pi / 180.00
            meta.addCV(
                CVAngle(atom_list[0], atom_list[1], atom_list[2], width, wall,
                        floor))

        elif cv_type == "dihedral":
            atom_list = get_atom_list(cv.atom)
            if len(atom_list) != 4:
                raise RuntimeError(
                    "Metadynamics: Dihedral-CV requires exactly 4 atoms/sites. You selected: "
                    + str(len(atom_list)))
            width = (cv['width'].val) * math.pi / 180.0
            if 'wall' in cv:
                wall = cv['wall'].val * math.pi / 180.00
            if 'floor' in cv:
                floor = cv['floor'].val * math.pi / 180.00
            meta.addCV(
                CVDihedral(atom_list[0], atom_list[1], atom_list[2],
                           atom_list[3], [width, width], wall, floor))
        elif cv_type in ["rmsd", "rmsd_alt"]:
            rmsd_weights = None
            superpos_weights = None
            atom_list = []
            for e in get_atom_list(cv.atom)[0]:
                atom_list.append(e)

            if len(atom_list) <= 3:
                raise RuntimeError(
                    "Metadynamics: RMSD-CV requires more than 3 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            ref_coords = []
            if cv_type == 'rmsd_alt':
                # use alternative coordinates set at setup, for rmsd_alt calculations
                for anum in atom_list:
                    ref_coords.append(model.atom[anum].property['r_altx'])
                    ref_coords.append(model.atom[anum].property['r_alty'])
                    ref_coords.append(model.atom[anum].property['r_altz'])
            else:
                for anum in atom_list:
                    ref_coords.extend(model.atom[anum].xyz)
            # use weights by setting r_mtd_rmsd_weight atom level property
            has_rmsd_prop = any([
                'r_mtd_rmsd_weight' in model.atom[i].property for i in atom_list
            ])
            has_superpos_prop = any([
                'r_mtd_superpos_weight' in model.atom[i].property
                for i in atom_list
            ])
            if (has_rmsd_prop and has_superpos_prop):
                rmsd_weights = []
                superpos_weights = []
                for anum in atom_list:
                    rmsd_weights.append(model.atom[anum].property.get(
                        "r_mtd_rmsd_weight", 0))
                    superpos_weights.append(model.atom[anum].property.get(
                        "r_mtd_superpos_weight", 0))
            elif (has_rmsd_prop or has_superpos_prop):
                raise RuntimeError(
                    "Metadyanamics:  If r_mtd_rmsd_weight is "
                    "defined then r_mtd_superpos_weight must be defined "
                    "as well")
            meta.addCV(
                CVrmsd(atom_list, ref_coords, width, rmsd_weights,
                       superpos_weights))
        elif cv_type == "rmsd_symm":
            atom_list = []
            for e in get_atom_list(cv.atom)[0]:
                #for e in get_atom_list(cv.atom):
                atom_list.append(e)
            if len(atom_list) <= 3:
                raise RuntimeError(
                    "Metadyamics: RMSD_SYMM-CV requires more than 3 atoms. You selected: "
                    + str(len(atom_list)))
            atom_list_combinations = get_local_symmetry(model, atom_list)
            width = cv['width'].val
            atom_map = {}
            # enumerate atom numbers
            sorted_atom_list = sorted(atom_list_combinations[0])
            for i, e in enumerate(atom_list_combinations[0]):
                atom_map[e] = i
            ref_coords_list = []
            for comb in atom_list_combinations:
                ref_coords = []
                for num in sorted_atom_list:
                    anum = comb[atom_map[num]]
                    ref_coords.append(model.atom[anum].x)
                    ref_coords.append(model.atom[anum].y)
                    ref_coords.append(model.atom[anum].z)
                ref_coords_list.append(ref_coords)
            meta.addCV(CVrmsd_symm(sorted_atom_list, ref_coords_list, width))
        elif cv_type == "zdist0":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 1:
                raise RuntimeError(
                    "Metadynamics: Zdist0-CV requires more than 1 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            meta.addCV(CVzdist0(atom_list, width))
        elif cv_type == "zdist":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 1:
                raise RuntimeError(
                    "Metadynamics: Zdist-CV requires more than 1 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            if 'wall' in cv:
                wall = cv['wall'].val
            if 'floor' in cv:
                floor = cv['floor'].val
            meta.addCV(CVzdist(atom_list, width, wall, floor))
        elif cv_type == "rgyr":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 2:
                raise RuntimeError(
                    "Metadynamics: Rgyr requires more than 2 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            meta.addCV(CVrgyr(atom_list, width))
        elif cv_type == "rgyr_mass":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 2:
                raise RuntimeError(
                    "Metadynamics: Rgyr requires more than 2 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            meta.addCV(CVrgyr_mass(atom_list, width))
        elif cv_type == "whim1":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 2:
                raise RuntimeError(
                    "Metadynamics: WHIM_1 requires more than 2 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            meta.addCV(CVwhim(atom_list, 1, width))
        elif cv_type == "whim2":
            atom_list = get_atom_list(cv.atom)[0]
            if len(atom_list) < 2:
                raise RuntimeError(
                    "Metadynamics: WHIM_1 requires more than 2 atom. You selected: "
                    + str(len(atom_list)))
            width = cv['width'].val
            meta.addCV(CVwhim(atom_list, 2, width))
        else:
            raise RuntimeError("unrecognized cv type: %s", cv_type)
    return meta


def generate_meta_cfg(meta_def, model):
    """
    Generate part of the config file for metadynamics simulation
    :param meta_def: The content of definition file for collective variables.
    :param model: topology file.
    :type meta_def: sea.Sea object
    :type model: cms.Cms object
    :return a string. Exception will be raised if encounting any errors.
    """
    meta = parse_meta(meta_def, model)
    return meta.generateCfg(model)


def get_meta_cfg_filename(meta_def):
    """
    Returns the name of the kerseq and cvseq files given a metadynamics
    definition file
    :param meta_def: The content of definition file for collective variables.
    :param model: topology file.
    """
    meta = parse_meta(meta_def, None)
    m = sea.Map(meta.generateCfg()).val
    kername = m['metadynamics_accumulators'][0]['name']
    cvname = m['name']

    return "force.term.ES = {metadynamics_accumulators.name = %s name = %s}" %\
        (kername, cvname)


class MetaDynamicsAnalysis:
    """
    Analysis tools for Desmond's metadynamics jobs. The class can be used
    and run from the command-line or as a module.
    """

    def __init__(self, data_fname, inp_fname=None, key=None):

        self.cutoff = 9.0
        self.height = []
        self.cv = []
        self.bins = []
        self.ranges = []
        self.time = []
        self.data = []

        if not inp_fname and not key:
            raise RuntimeError('Either the frontend .cfg file or a sea.Map key '
                               'must be provided')

        if key is None:
            self._parseInp(inp_fname)
        else:
            self._getInp(key)

        self._parseData(data_fname)

    def _parseInp(self, inp_fname):
        try:
            fh = open(inp_fname, "r")
            s = fh.read()
            fh.close()
        except IOError:
            raise IOError(
                "Reading the metadynamics input file failed: %s" % inp_fname)

        try:
            key = sea.Map(s)
        except ValueError:
            raise IOError(
                "Parsing the metadynamics input file failed: %s" % inp_fname)

        # This allows the parsing of -in.cfg...
        if hasattr(key, 'meta'):
            self._getInp(key.meta)
        # ...and -out.cfg files
        else:
            self._getInp(key.ORIG_CFG.meta)

    def _getInp(self, meta):
        try:
            self.cutoff = meta.cutoff.val
        except:
            self.cutoff = 9.0

        for r in meta.cv:
            self.cv.append(r.type.val)
            try:
                self.bins.append(r.nbin.val)
            except:
                self.bins.append(21)
            name = r.type.val

            try:
                cv_range = r.range.val
                if ([] == cv_range):
                    raise AttributeError()
                if (name in ('angle', 'dihedral')):
                    cv_range = [
                        cv_range[0] * math.pi / 180.0,
                        cv_range[1] * math.pi / 180.0
                    ]
                self.ranges.append(cv_range)
            except AttributeError:
                if name == 'angle':
                    self.ranges.append([0, math.pi])
                elif name == 'dihedral':
                    self.ranges.append([-math.pi, math.pi])
                else:
                    self.ranges.append([])

    def _parseData(self, data_fname):
        try:
            fh = open(data_fname, "r")
            s = fh.readlines()
            fh.close()
        except IOError:
            raise IOError(
                "Reading the metadynamics data file failed: %s" % data_fname)
        rc = self.cv
        data = []
        for line in s:
            l = line.split()
            if len(l) == 0 or l[0] == '#':
                continue
            if (float(l[1]) == 0.0):
                continue
            self.time.append(float(l[0]))
            self.height.append(float(l[1]))
            row_rc = []
            n = 2
            simple_cvs = [
                'dist', 'rmsd', 'rmsd_alt', 'zdist0', 'zdist', 'rmsd_symm',
                'rgyr', 'rgyr_mass', 'whim1', 'whim2'
            ]
            for r in rc:
                each_rc = []
                if r in simple_cvs:
                    each_rc.append(float(l[n]))
                    each_rc.append(float(l[n + 1]))
                    n += 2
                elif r == 'angle':
                    # angle is reported in kernel file as cos(angle), converted back to radians
                    each_rc.append(math.acos(float(l[n])))
                    each_rc.append(float(l[n + 1]))
                    n += 2
                elif r == 'dihedral':
                    # dihedrals are reported in kernel file as combination of two CVs cos(angle)
                    # and sin(angle), here we compress them back to one CV
                    cosphi = float(l[n])
                    wcosphi = float(l[n + 1])
                    sinphi = float(l[n + 2])
                    #if ((cosphi <= 0.0) & (sinphi <= 0.0)): phi = -1*math.acos(cosphi)
                    #if ((cosphi <= 0.0) & (sinphi > 0.0)): phi = math.acos(cosphi)
                    #if ((cosphi > 0.0) & (sinphi <= 0.0)): phi = -1*math.acos(cosphi)
                    #if ((cosphi > 0.0) & (sinphi > 0.0)): phi = math.acos(cosphi)
                    phi = math.atan2(sinphi, cosphi)
                    each_rc.append(phi)
                    each_rc.append(wcosphi)
                    n += 4
                row_rc.append(each_rc)
            data.append(row_rc)

        self.time = numpy.array(self.time)
        self.height = numpy.array(self.height)

        data = numpy.array(data)
        self.centers = data[:, :, 0]
        self.scales = numpy.reciprocal(data[:, :, 1])

    def evaluate(self, x):
        h = self.height
        centers = self.centers
        scales = self.scales
        nker = centers.shape[0]

        dihedral = []
        non_dihedral = []
        rc = self.cv
        for i, r in enumerate(rc):
            if r == 'dihedral':
                dihedral.append(i)
            else:
                non_dihedral.append(i)

        c2 = self.cutoff * self.cutoff
        pe = 0.0
        for c in range(nker):
            z2 = 0.0
            for i in non_dihedral:
                v = (x[i] - centers[c, i]) * scales[c, i]
                z2 += v * v
            if z2 >= c2:
                continue
            for i in dihedral:
                v = (math.cos(x[i]) - math.cos(centers[c, i])) * scales[c, i]
                z2 += v * v
                v = (math.sin(x[i]) - math.sin(centers[c, i])) * scales[c, i]
                z2 += v * v
            if z2 < c2:
                pe += h[c] * math.exp(-0.5 * z2)

        return -pe

    def computeFES(self, out_fname='', units='degrees', progress_callback=None):
        """
        This function figures out the grid from the ranges and the bins given
        the cfg.  For each gaussian, add it to the previous gaussian sum for each
        grid point.
        """
        # Units the data is in ('radians' or 'degrees')
        if units not in ['radians', 'degrees']:
            raise RuntimeError('Invalid "units" supplied. Choices are: '
                               '"radians", "degrees"')

        # the range for dist-cv is dynamically set
        for i in range(len(self.ranges)):
            if not self.ranges[i]:
                self.ranges[i] = [
                    self.centers[:, i].min(), self.centers[:, i].max()
                ]

        shape = tuple(self.bins)
        edges = []
        # generate edges for FES calculation
        for i in range(len(self.bins)):
            r = old_div((self.ranges[i][1] - self.ranges[i][0]),
                        (self.bins[i] - 1))
            edges.append(
                numpy.arange(self.ranges[i][0], self.ranges[i][1] + r, r))

        FES = numpy.zeros(shape)
        data = []

        total_steps = numpy.cumprod(shape)[-1]
        step = 0
        interval = int(old_div(total_steps, 100))  # 1% intervals
        if interval == 0:
            interval = 1

        backend = get_backend()

        if backend:
            # Setting the JC backend callback to "progress_callback" will allow
            # jobs in JC to be monitored as well as jobs that are being run in
            # a thread to be given updates
            progress_callback = backend.setJobProgress
            progress_callback(description="Calculating free energy surface")

        if progress_callback:
            progress_callback(step, total_steps)

        for idx in numpy.ndindex(shape):
            x = []
            for i in range(len(idx)):
                x.append(edges[i][idx[i]])

            FES[idx] = self.evaluate(x)
            x.append(FES[idx])

            # Convert the data values to degrees if units specified are degrees
            # This needs to happen after the evaluate call b/c evaluate expects
            # the values to be in radians
            angstrom_units = [
                'dist', 'rmsd', 'rmsd_alt', 'rmsd_symm', 'zdist', 'zdist0',
                'rgyr', 'rgyr_mass', 'whim1', 'whim2', 'whim3'
            ]
            if units == 'degrees':
                for i, cv in enumerate(self.cv):
                    if cv in angstrom_units:
                        continue
                    else:
                        x[i] = numpy.degrees(x[i])

            data.append(x)

            step += 1

            if progress_callback:
                # Insure updates only happen at 5% intervals
                if step % interval == 0:
                    progress_callback(step, total_steps)

        # Make sure to get the last update too
        if progress_callback:
            progress_callback(step, total_steps)

        if out_fname:
            self.writeFES(out_fname, data, self.bins, self.cv, units)
            if backend:
                # If running under job control
                backend.addOutputFile(out_fname)

        return (data, units)

    @staticmethod
    def convertDataToPlot(bins, data):
        """
        Converts data, usually read in from an exported plot result, to
        structures usable by the plot.

        :param bins: The FES shape
        :type  bins: list or tuple
        :param data: List of lists containing cv and FES values

        :returns: list of x and y values
        :returns: array of FES

        """
        bins = tuple(bins)
        blen = bins[0]

        edges = []
        FES = None

        # If the length of the bins is 1 we have a line plot
        if len(bins) == 1:
            edges.append(numpy.array([d[0] for d in data]))
            FES = numpy.array([d[1] for d in data])
        # If the length of the bins is 2 we have a colormesh and the
        # data needs to be parsed differently
        elif len(bins) == 2:
            # This gets the first value from every i'th array, where i
            # is the length of the bin
            edges.append(numpy.array([d[0] for d in data[::blen]]))
            # This gets the 2nd value from the 0th array to the i'th
            # array, where i is the length of the bin
            edges.append(numpy.array([d[1] for d in data[0:blen]]))
            FES = numpy.zeros(bins)

            for i, d in enumerate(data):
                r = i % blen
                s = old_div(i, blen)
                #BUGFIX
                FES[r][s] = d[-1]

        return (edges, FES)

    @staticmethod
    def convertPlotToData(bins, edges, FES):
        """
        Takes data used to plot FES values and converts it to a list of lists
        for exporting purposes.

        :param  bins: The FES shape
        :type   bins: list or tuple
        :param edges: The x and y values for the plot
        :type  edges: List of lists
        :param   FES: The FES values for the plot
        :type    FES: `numpy.array`

        :returns: List of lists containing cv and FES values

        """
        bins = tuple(bins)
        nbins = len(bins)

        data = []
        for idx in numpy.ndindex(bins):
            x = []
            for i in range(len(idx)):
                x.append(edges[i][idx[i]])
            if nbins == 2:
                fes_idx = (idx[1], idx[0])
            else:
                fes_idx = idx[0]
            x.append(FES[fes_idx])
            data.append(x)

        return data

    @staticmethod
    def writeFES(fname, data, fes_shape, cvs, units):
        """
        Write out the free energy distribution in a common way. The
        GUI utilizing this class needs to write out `data` from
        `self.computeFES`
        """
        header = '# Desmond Metadynamics - Free Energy Surface\n'
        header += '# FES shape: %s\n' % ' '.join(str(b) for b in fes_shape)
        header += '# FES units: %s\n' % units
        header += '# %s free_energy\n\n' % ' '.join(cvs)

        f = open(fname, "w")

        f.write(header)

        for x in data:
            f.write('%s\n' % ' '.join(str(f) for f in x))

        f.close()

        return fname


def read_meta_cfg(config, model):
    """
    Read config file for metadynamics simulation
    :param meta_def: The content of definition file for collective variables.
    :param model: topology file.
    :type meta_def: string
    :type model: cms.Cms object
    :return a Meta
    """

    try:
        fh = open(config, "r")
        s = fh.read()
        fh.close()
    except IOError:
        raise IOError("Reading the metadynamics input file failed: %s" % config)
    try:
        key = sea.Map(s)
    except ValueError:
        raise IOError("Parsing the metadynamics input file failed: %s" % config)
    try:
        m = parse_meta(key.meta, model)
    except:
        m = parse_meta(key.ORIG_CFG.meta, model)

    m.generateCfg(model)


def get_distance(model, atom_list):
    """
    Check distance of the two groups of atoms defined in the atom list
    :param model: topology file
    :type model: cms.Cms object
    :param atom_list: atom list
    :type atom_list: list
    """

    from schrodinger.application.desmond.packages.analysis import Pbc
    import math

    if isinstance(atom_list[0], int):
        atom = model.atom[atom_list[0]]
        p1 = numpy.array([atom.x, atom.y, atom.z])
    else:
        from schrodinger.structutils import analyze
        p1 = analyze.center_of_mass(model, atom_list[0])
    if isinstance(atom_list[1], int):
        atom = model.atom[atom_list[1]]
        p2 = numpy.array([atom.x, atom.y, atom.z])
    else:
        from schrodinger.structutils import analyze
        p2 = analyze.center_of_mass(model, atom_list[1])

    box = numpy.reshape(
        numpy.array([model.property[prop] for prop in SIM_BOX]), (3, 3))
    bc = Pbc(box)
    diff = bc.calcMinimumDiff(p1, p2)
    distance = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    return distance


if __name__ == '__main__':
    from schrodinger.utils import cmdline

    usage = '''
  %prog -i meta.inp -d kerseq -o output

Description:
  %prog computes free energy distribution
'''

    cmdline = cmdline.SingleDashOptionParser(usage=usage, version='%prog 1.0')

    cmdline.add_option(
        '-i', dest='input', help="meta dynamics input cfg file name")
    cmdline.add_option(
        '-d', dest='data', help="meta data file name (kerseq file)")
    cmdline.add_option('-o', dest='output', help="output file name")
    cmdline.add_option('-m', dest='model', help="input cms file")

    opts, args = cmdline.parse_args()
    if not opts.input:
        print("Error: input file must be specified, use -h for help.")
        sys.exit(1)

    #meta = MetaDynamicsAnalysis(opts.data,inp_fname=opts.input )
    #print meta.evaluate([1.0, 0.1, 0.1])
    #meta.computeFES(opts.output)
    if (opts.input and opts.model):
        read_meta_cfg(opts.input, Cms(opts.model))

    elif (opts.input and opts.data and opts.output):
        meta = MetaDynamicsAnalysis(opts.data, inp_fname=opts.input)
        meta.computeFES(opts.output)


def get_local_symmetry(st, atom_list):
    # get hydrogens that are attached to the atoms of interest (needed for substructure matching)
    import schrodinger.structutils as structutils
    a = [str(e) for e in atom_list]
    asl_list_str = ', '.join(a)
    asl_select = '((withinbonds 1 (a.num  ' + asl_list_str + ')) and (atom.ele H)) or (a.num ' + asl_list_str + ' )'

    st_indexes = structutils.analyze.evaluate_asl(st, asl_select)
    st_temp = st.extract(st_indexes, copy_props=True)

    coo_smarts = '[C-0X3,N+X3](=[O-0X1])[O-X1]'  #carboxylate oxigens or nitro (-NO2) group
    coo_frags = structutils.analyze.evaluate_smarts(
        st, coo_smarts, first_match_only=False, unique_sets=True)
    if len(coo_frags):
        print("\t", len(coo_frags),
              " Carboxylate or nitro fragment(s) detected")
    for i in coo_frags:
        st_temp.atom[i[1]].formal_charge = -1
        st_temp.addBond(i[0], i[1], 1)

    soo_smarts = '[#6,#7][S-0X4](=[O-0X1])(=[O-0X1])[#6,#7]'  #sulfonyl oxygens
    soo_frags = structutils.analyze.evaluate_smarts(
        st, soo_smarts, first_match_only=False, unique_sets=True)
    if len(soo_frags):
        print("\t", len(soo_frags), " Sulfonyl fragment(s) detected")
    for i in soo_frags:
        st_temp.atom[i[2]].formal_charge = -1
        st_temp.addBond(i[1], i[2], 1)

    piperazine_smarts = '[!#1][C,N,O]1[C-0X4][C-0X4][C,N,O]([C-0X4])[C-0X4][!#1]1'  # 2f4j.pdb piperazine
    piperazine_frags = structutils.analyze.evaluate_smarts(
        st, piperazine_smarts, first_match_only=False, unique_sets=True)
    if len(piperazine_frags):
        print("\t", len(piperazine_frags),
              " Piperazine-like fragment(s) detected")
    for i in piperazine_frags:
        st_temp.addBond(i[2], i[3], 2)

    piperdin_smarts = '[!#1][N,C]1[C-0X4][C-0X4][C,N,O][C-0X4][C-0X4]1'  #{2bfy,1i91,1sj0}.pdb piperdin
    piperdin_frags = structutils.analyze.evaluate_smarts(
        st, piperdin_smarts, first_match_only=False, unique_sets=True)
    if len(piperdin_frags):
        print("\t", len(piperdin_frags), " Piperdin-like  fragment(s) detected")
    for i in piperdin_frags:
        st_temp.addBond(i[2], i[3], 2)

    cyclopropane_smarts = '[#6,#7,#8][C-0X4]1[C-0X4][C-0X4]1'  #cyclopropane
    cyclopropane_frags = structutils.analyze.evaluate_smarts(
        st, cyclopropane_smarts, first_match_only=False, unique_sets=True)
    if len(cyclopropane_frags):
        print("\t", len(cyclopropane_frags),
              " Cyclopropane fragment(s) detected")
    for i in cyclopropane_frags:
        st_temp.addBond(i[1], i[2], 2)

    dimethylamino_smarts = '[#1][C,N]([C-0X4]([#1])([#1])[#1])[C-0X4]([#1])([#1])[#1]'  #dimethyl {amino
    dimethylamino_frags = structutils.analyze.evaluate_smarts(
        st, dimethylamino_smarts, first_match_only=False, unique_sets=True)
    if len(dimethylamino_frags):
        print("\t", len(dimethylamino_frags),
              " Dimethylamino fragment(s) detected")
    for i in dimethylamino_frags:
        st_temp.addBond(i[1], i[6], 2)

    dimethylether_smarts = '[#8,#16]([C-0X4]([#1])([#1])[#1])[C-0X4]([#1])([#1])[#1]'  #dimethyl {ether or sulfide}
    dimethylether_frags = structutils.analyze.evaluate_smarts(
        st, dimethylether_smarts, first_match_only=False, unique_sets=True)
    if len(dimethylether_frags):
        print("\t", len(dimethylether_frags),
              " Dimethylether or sulfide fragment(s) detected")
    for i in dimethylether_frags:
        st_temp.addBond(i[0], i[5], 2)

    difluoro_smarts = '[#6,#7,#8][C]([F])([F])[!#9]'  #difluoro group 1xoq, 1mu6
    difluoro_frags = structutils.analyze.evaluate_smarts(
        st, difluoro_smarts, first_match_only=False, unique_sets=True)
    if len(difluoro_frags):
        print("\t", len(difluoro_frags), " Difluoro fragment(s) detected")
    for i in difluoro_frags:
        st_temp.addBond(i[1], i[3], 2)

    azetin_smarts = '[!#1][C-OX4,N+X4]1[C-0X4][C-0X4]([!#1])[C-0X4,N+X4]1'  # azetin-like  2chm.pdb
    azetin_frags = structutils.analyze.evaluate_smarts(
        st, azetin_smarts, first_match_only=False, unique_sets=True)
    if len(azetin_frags):
        print("\t", len(azetin_frags), " Azetin-like fragment(s) detected")
    for i in azetin_frags:
        st_temp.addBond(i[2], i[3], 2)

    amino_smarts = '[#1][N+X3](=[C-0X3]([!#1])[N-0X3]([#1])[#1])[#1]'  #amino group 1o30
    amino_frags = structutils.analyze.evaluate_smarts(
        st, amino_smarts, first_match_only=False, unique_sets=True)
    if len(amino_frags):
        print("\t", len(amino_frags), " Amino fragment(s) detected")
    for i in amino_frags:
        st_temp.addBond(i[1], i[2], 1)
        st_temp.atom[i[1]].formal_charge = 0

    cyclopentyl_smarts = '[!#1][!#1]1[C-0X4][C-0X4][C-0X4][C-0X4]1'  #cyclopentyl (5-member rings carbon) 1o2h.pdb
    cyclopentyl_frags = structutils.analyze.evaluate_smarts(
        st, cyclopentyl_smarts, first_match_only=False, unique_sets=True)
    if len(cyclopentyl_frags):
        print("\t", len(cyclopentyl_frags), " Cyclopentyl ring(s) detected")
    for i in cyclopentyl_frags:
        st_temp.addBond(i[1], i[2], 2)

    cycloheptane_smarts = '[#6,#7,#8]1[C-0X4][C-0X4][c-0X3][c-0X3][C-0X4][C-0X4]1'  #cyclopheptane (7-member ring) 1ttm
    cycloheptane_frags = structutils.analyze.evaluate_smarts(
        st, cycloheptane_smarts, first_match_only=False, unique_sets=True)
    if len(cycloheptane_frags):
        print("\t", len(cycloheptane_frags), " Cycloheptane ring(s) detected")
    for i in cycloheptane_frags:
        st_temp.addBond(i[0], i[1], 2)

    sulfamate_smarts = '[!#1][O-0X2][S-0X4](=[O-0X1])(=[O-0X1])[#6,#7,#8]'  #ulfamate 1ttm
    sulfamate_frags = structutils.analyze.evaluate_smarts(
        st, sulfamate_smarts, first_match_only=False, unique_sets=True)
    if len(sulfamate_frags):
        print("\t", len(sulfamate_frags), " Sulfamate ring(s) detected")
    for i in sulfamate_frags:
        st_temp.addBond(i[2], i[3], 1)

    pyrazole_smarts = '[!#1][c-0X3]1[c-0X3][n][n][c-0X3]1'  #pyrazole, 1b2y
    pyrazole_frags = structutils.analyze.evaluate_smarts(
        st, pyrazole_smarts, first_match_only=False, unique_sets=True)
    if len(pyrazole_frags):
        print("\t", len(pyrazole_frags), " Pyrazole ring(s) detected")
    for i in pyrazole_frags:
        st_temp.atom[i[2]].formal_charge = 1

    pyrazole_smarts = '[!#1][c-0X3]1[n][c-0X3][c-0X3][n]1'  #imidazole,  1c1u
    pyrazole_frags = structutils.analyze.evaluate_smarts(
        st, pyrazole_smarts, first_match_only=False, unique_sets=True)
    if len(pyrazole_frags):
        print("\t", len(pyrazole_frags), " Imidazole ring(s) detected")
    for i in pyrazole_frags:
        st_temp.atom[i[2]].formal_charge = 1

    heavy_atom_list = structutils.analyze.evaluate_asl(st_temp,
                                                       'all and not a.e H')
    #structutils.build.delete_hydrogens(st_temp)
    st_temp_smarts = structutils.analyze.generate_smarts(
        st_temp, atom_subset=heavy_atom_list)
    st_temp_list = structutils.analyze.evaluate_smarts(
        st_temp, st_temp_smarts, first_match_only=False, unique_sets=False)
    return st_temp_list
