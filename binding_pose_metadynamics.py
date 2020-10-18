import argparse
import csv

import itertools
import math
import os
import re
import shutil
import sys
import tarfile
import textwrap
import warnings
import subprocess
from past.utils import old_div

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import schrodinger.application.desmond.cms as cms
import schrodinger.application.desmond.packages.topo as topo
import schrodinger.application.desmond.packages.traj as traj
import schrodinger.application.prime.packages.PoseScoring as ps
import schrodinger.application.prime.packages.utilities as psp_util
from schrodinger import structure
from schrodinger.application.prime.packages import Prime
from schrodinger.application.prime.packages import executeparallel
from schrodinger.application.prime.packages import primeparser
from schrodinger.application.prime.packages import structurealign
from schrodinger.infra import mm
from schrodinger.job import jobcontrol
from schrodinger.job import queue
from schrodinger.structutils import analyze
from schrodinger.structutils import rmsd
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.rmsd import calculate_in_place_rmsd
from schrodinger.utils.subprocess import check_output
from schrodinger.application.prime.packages.ifd_plus_stages.stage import Stage

with warnings.catch_warnings():
    # We do not include openpyxl in our schrodinger distribution
    warnings.filterwarnings(
        "ignore", message="Installed openpyxl is not supported at this time.")
    import pandas as pd

usage = """
Metadynamics Analysis of Binding Poses
$SCHRODINGER/run -FROM psp %s <input complexes> -HOST <CPU Queue> -SUBHOST <GPU Queue>

This algorithm is designed to rank various binding configurations of a single
ligand in a single binding site.

The metadynamics portion of the dynamics will be returned as well as a plot with
the extension -out.pdf which plots the average value of the collective variable
over the simulation

Output Properties:
    r_psp_MetadynamicsBinding_CompScore -- Composite Score linearly combining
        the r_psp_MetadynamicsBinding_PoseScore and
        r_psp_MetadynamicsBinding_Persistence scores.  Lower values equate to
        more stable complexes.  The equation is PoseScore - 5 * Persistence
    r_psp_MetadynamicsBinding_PoseScore -- Expectation of the RMSD of the
        pose over the course of the metadynamics.  Lower values equate to
        more stable complexes.
    r_psp_MetadynamicsBinding_Persistence -- Average persistence of contacts
        over the course of the metadynamics trajectories.  Higher values equate
        to more stable complexes
    r_psp_MetadynamicsBinding_Persistence_HBond -- Average persistence of
        HBonds
    r_psp_MetadynamicsBinding_Persistence_Pi -- Average persistence of Pi-Pi
        stacking and T-Shaped interactions
    r_psp_MetadynamicsBinding_In_RMSD -- Optional RMSD after the relaxation
       step.  This can help identify cases where the relaxation before the
       metadynamics is materially changing the pose

""" % (os.path.basename(sys.argv[0]))

version = "BindingPoseMetadynamics v1.0"


def moving_average(x, n):
    """
    For a given array x, return the moving average with a window
    of 2*n+1 elements centered.
    x {list or numpy array}
    n {integer}
    Output is an numpy array with the same size as x
    """
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.mean(x[max(0, i - n):min(len(x), i + n + 1)])
    return y


def split_path(fn):
    """
    For a given output filename, split it into an array of entries
    using the os.path.split to define where to split.
    fn {string, path} input filename to split
    return value {list of strings}
    """
    output = []
    while (fn):
        split_fn = os.path.split(fn)
        output = [split_fn[1]] + output[:]
        fn = split_fn[0]
    return output


class CreateSystemJob(queue.JobControlJob):
    """
    Subclass of a JobcontrolJob class that will create solvated desmond
    systems from maestro files.
    """
    TEMPLATE_SET_FAMILY = \
            """task {
              set_family = {
                 desmond = {
                    checkpt = {
                       write_last_step = false
                    }
                 }
                 simulate = {
                    trajectory = {
                       center = "solute"
                    }
                 }
              }
              #task = "desmond:fep"
              task = "desmond:auto"
            }"""
    TEMPLATE_SET_FAMILY = textwrap.dedent(TEMPLATE_SET_FAMILY)

    TEMPLATE_BUILD_GEOMETRY_SOLUBLE = \
            """build_geometry {
                  box_shape = "orthorhombic"
                  override_forcefield = %(forcefield)s
                  buffer_width = 10.0
                  neutralize_system = true
                  rezero_system = true
                }"""
    TEMPLATE_BUILD_GEOMETRY_SOLUBLE = textwrap.dedent(
        TEMPLATE_BUILD_GEOMETRY_SOLUBLE)

    TEMPLATE_ASSIGN_FORCEFIELD = \
            """assign_forcefield {
              %(hmr_option)s
              forcefield = %(forcefield)s
            }"""
    TEMPLATE_ASSIGN_FORCEFIELD = textwrap.dedent(TEMPLATE_ASSIGN_FORCEFIELD)

    def __init__(self, ipose, ct, no_hmr, **kwargs):
        """
            :type ipose: int
            :param ipose: The ipose (pose number) for the CreateSystemJob.
                          This is one-based.

            :type ct: schrodinger.Structure
            :param ct: Input structure to work with.  This
            structure should not include solvents or membranes

            :type no_hmr: bool
            :param no_hmr: Whether or not
            structure should not include solvents or membranes

        All other keyword options are passed through to the underlying
        JobcontrolJob class, but command_dir also need to be processed
        here to create the temporary files in the corect locations
        """

        self.command_dir = kwargs.get("command_dir")

        self.out_cms_path, self.msj_path, self.log_path,\
                self.input_structure_path = self.files_to_save(ipose, self.command_dir)

        jobname = self.get_jobname(ipose)

        ct.write(self.input_structure_path)

        # If we ever want to give the user
        # the option to change the forcefield, it would
        # be through here.
        self.forcefield = mm.mmffld_get_ffld_name(
            psp_util.get_default_force_field())
        self.no_hmr = no_hmr

        self.composeMsj(self.msj_path)

        cmd = [
            os.path.join(os.getenv("SCHRODINGER"), "utilities", "multisim"),
            "-JOBNAME",
            jobname,
            "-m",
            os.path.basename(self.msj_path),
            os.path.basename(self.input_structure_path),
            "-o",
            os.path.basename(self.out_cms_path),
            "-maxjob",
            "1",
            '-lic',
            "DESMOND_GPGPU:16",
        ]

        kwargs['name'] = jobname
        queue.JobControlJob.__init__(self, cmd, **kwargs)

    @classmethod
    def files_to_save(cls, ipose, command_dir=None):
        """
            Returns a list of files that should be saved
            from the SystemBuilder jobs.

            :type ipose: int
            :param ipose: The ipose (pose number) for the CreateSystemJob.
                          This is one-based.

            :type command_dir: str
            :param command_dir: An optional directory where all the files
                                will reside.

            :rtype: list of str
            :return: Returns a list of files to save in the following
                     order:
                        out_cms_path, msj_path, log_path, input_structure_path
        """
        jobname = cls.get_jobname(ipose)

        out_cms_filename = jobname + "-out.cms"
        msj_path = jobname + '.msj'
        log_path = jobname + "_multisim.log"
        input_structure_path = jobname + "-in.maegz"

        files_to_save = [
            out_cms_filename, msj_path, log_path, input_structure_path
        ]

        if command_dir:
            files_to_save = [
                os.path.join(command_dir, file_to_save)
                for file_to_save in files_to_save
            ]

        return files_to_save

    @staticmethod
    def get_jobname(ipose):
        """
            :type ipose: int
            :param ipose: The ipose (pose number) for the CreateSystemJob.
                          This is one-based.

            :rtype: str
            :return: The unique jobname for thie given ipose.
        """
        id = Stage.pose_id(ipose, abbrev=True)
        jobname = "SystemBuilder_{}".format(id)

        return jobname

    @classmethod
    def template_assign_forcefield(cls, forcefield, no_hmr):

        keywords = {
            'forcefield': forcefield,
        }

        if no_hmr:
            keywords['hmr_option'] = ''
        else:
            keywords['hmr_option'] = 'hydrogen_mass_repartition = true'

        msj_section = cls.TEMPLATE_ASSIGN_FORCEFIELD % (keywords)

        return msj_section

    def getPath(self, filename):
        """
        Helper function that returns the path of a file that is in the
        command_dir if a command_dir was provided or in the
        current working directory otherwise
        inputs
        filename { string } name of the file
        return values {string} path to that file in the command_dir specified
            during class initialization
        """
        if self.command_dir is not None:
            return os.path.join(self.command_dir, filename)
        return filename

    def composeMsj(self, msj_path):
        """
        Create an MSJ file that will be run by Desmond.
        :type msj_path: str
        :param msj_path: Path to msj file to be created.
        """
        build_geometry_template = self.TEMPLATE_BUILD_GEOMETRY_SOLUBLE
        template = "\n\n".join([
            self.TEMPLATE_SET_FAMILY, build_geometry_template,
            self.template_assign_forcefield(self.forcefield, self.no_hmr)
        ])
        values = {}
        values['forcefield'] = self.forcefield
        with open(msj_path, "wt") as fp:
            fp.write(template % values)


class BindingPoseMetadynamics(executeparallel.ExecuteParallel):
    """
    Run metadynamics on each pose using the
    aligned ligand RMSD as the collective variable and then plot
    that collective variable
    """
    POSE_TAG = "i_psp_BindingPoseMetadynamics_Pose_Number"
    TRIAL_TAG = "i_psp_BindingPoseMetadynamics_Trial_Number"
    COLORS = [
        'b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'b--', 'g--', 'r--', 'c--',
        'm--', 'y--', 'k--', 'b:', 'g:', 'r:', 'c:', 'm:', 'y:', 'k:'
    ]
    ERROR_BAR_COLORS = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k',
        'b', 'g', 'r', 'c', 'm', 'y', 'k'
    ]
    MTD_MSJ_FILENAME = 'binding_pose_mtd.msj'
    PRODUCTION_STAGE = 8  # MtD Production Stage Number
    GCMC_LIG_PROPERTY = 'i_gcmc_ligand'  # Identifes the GCMC region
    INTERNAL_LIG_PROPERTY = 'b_psp_metadynamics_lig_atom'
    ORIGINAL_ATOM_BOOL = 'b_psp_metadynamics_orig_atom'  #for tracking the original atoms in the input ct
    INTERNAL_LIG_ASL = '(atom.{})'.format(INTERNAL_LIG_PROPERTY)
    TGZ_ALIGN_ASL = '(atom.ptype "%s" OR atom.ptype "%s")' % (" CA ", " C5'")
    KTEMP = 4  # Well tempering parameter
    ORIGINAL_TITLE_PROPERTY = "s_psp_MetadynamicsBinding_Original_Title"
    ORIGINAL_GROUP_TITLE_PROPERTY = "s_psp_MetadynamicsBinding_Original_Group_Title"
    ORIGINAL_GROUP_ID_PROPERTY = "s_psp_MetadynamicsBinding_Original_Group_Id"
    ORIGINAL_GROUP_COLLAPSED_PROPERTY = "b_psp_MetadynamicsBinding_Original_Group_Collapsed"
    GROUP_TITLE_PROPERTY = "s_m_subgroup_title"
    GROUP_ID_PROPERTY = "s_m_subgroupid"

    DEFAULT_HEIGHT = 0.05
    DEFAULT_CV = "ligand_in_site"
    DEFAULT_SCORE_FRACTION = 0.2
    DEFAULT_LABEL_PROPERTY = "r_ifdreport_ligand_RMSD"
    DEFAULT_PROD_TIME = 10  # ns

    GROUP_PROPERTY_PAIRS = list(
        zip([
            ORIGINAL_GROUP_TITLE_PROPERTY,
            ORIGINAL_GROUP_ID_PROPERTY,
        ], [GROUP_TITLE_PROPERTY, GROUP_ID_PROPERTY]))

    SKA_RES_NAME = "s_psp_ska_residue_name"
    MAX_GAUSSIAN_CENTER = 30
    MAX_GAUSSIAN_WIDTH = 1
    MAX_JOBNAME_BASE_LEN = 6
    METADIR_POSTFIX = 'metadynamics'

    def __init__(self, script_fn):
        executeparallel.ExecuteParallel.__init__(
            self,
            script_fn,
            usage, ["-out.csv", "-out.png", "-out.pdf", "-out.maegz"],
            use_jobdj="manual",
            uses_subhost=True,
            version=version)

    @classmethod
    def get_multisim_job_base(cls, mtd_jobname):
        multisim_jobname = mtd_jobname + "_" * cls.MAX_JOBNAME_BASE_LEN
        multisim_jobname = multisim_jobname[:cls.MAX_JOBNAME_BASE_LEN]
        return multisim_jobname

    @classmethod
    def get_multisim_jobname(cls, mtd_jobname, ipose, itrial):
        """Given the jobname for the entire binding pose
           metadynamics job, as well as the ipose and itrial,
           will return the jobname that will be used for
           multisim jobs.

           The ipose and itrial should be zero-based.

           These are all relative to the trial dir.
        """
        multisim_jobname = cls.get_multisim_job_base(mtd_jobname)
        pose_id = Stage.pose_id(ipose + 1, abbrev=True)
        multisim_jobname = "{}_{}_{:02d}".format(multisim_jobname, pose_id,
                                                 itrial + 1)
        return multisim_jobname

    @staticmethod
    def metadir_basename(opts):
        if opts.metadir_basename:
            meta_dir = opts.metadir_basename
        else:
            meta_dir = opts.jobname
        return meta_dir

    @classmethod
    def get_job_dir(cls, mtd_jobname):
        job_dir = '{}_{}'.format(mtd_jobname, cls.METADIR_POSTFIX)
        return job_dir

    @classmethod
    def get_trial_dir(cls, mtd_jobname, ipose, itrial):
        pose_dir = cls.get_pose_dir(mtd_jobname, ipose)
        trial_dir = 'trial_{:02d}'.format(itrial + 1)
        trial_dir = os.path.join(pose_dir, trial_dir)
        return trial_dir

    @classmethod
    def get_pose_dir(cls, mtd_jobname, ipose):
        job_dir = cls.get_job_dir(mtd_jobname)
        pose_dir = os.path.join(job_dir, Stage.pose_id(ipose + 1))
        return pose_dir

    @classmethod
    def get_multisim_files_to_save(cls, multisim_jobname, run_mode):
        """Returns a list of files that should be saved from each
           multisim job
        """

        files_to_save = \
                [cls.MTD_MSJ_FILENAME,
                 multisim_jobname + '_multisim.log',
                 multisim_jobname + '-multisim_checkpoint']
        if run_mode == 'md':
            # Must grab the output tgz two stages back.
            # One stage back is the 'stop' stage.
            files_to_save.append('{}_{}-out.tgz'.format(
                multisim_jobname, cls.PRODUCTION_STAGE - 2))
        elif run_mode == "system_builder":
            pass
        else:
            files_to_save.append('{}_{}-out.tgz'.format(multisim_jobname,
                                                        cls.PRODUCTION_STAGE))
        return files_to_save

    def addParserArguments(self, parser):
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.add_argument(
            "struct_file",
            nargs="+",
            action=primeparser.StoreFile,
            help="Structures to process.  These should be Ligand-Receptor\n" +
            "complex mae files or pre-equilibrated CMS files.  The most\n" +
            "common source for the complex mae files will be the output\n" +
            "of IFD runs.  Equilibrated CMS files are designed to be\n" +
            "used for systems containing membranes, nucleic acids or\n" +
            "any other components that require additional or different\n" +
            "equilibration protocols than the default 'relaxation'\n" +
            "protocol provided by Desmond which has been designed\n" +
            "for soluble proteins.")
        parser.add_argument(
            "-ligand_asl",
            required=True,
            action=primeparser.StoreAsl,
            help='ASL Expression defining the ligand (required).\n'
            'example "res.ptype UNK"')
        # md_host and md_nproc should be replaced by -SUBHOST HOST:NPROC
        md_host_msg ="The -md_host and -md_nproc arguments have been removed.\n"+\
               " Use -SUBHOST instead to specify nodes capable of running\n"+\
               "GPU Desmond"
        parser.add_argument(
            "-md_host",
            action=primeparser.display_message_and_exit(md_host_msg),
            help=argparse.SUPPRESS)
        parser.add_argument(
            "-md_nproc",
            action=primeparser.display_message_and_exit(md_host_msg),
            help=argparse.SUPPRESS)
        # md_host and md_nproc should be replace by -SUBHOST HOST:NPROC
        parser.add_argument(
            "-dev_test_jaco",
            default=False,
            action="store_true",
            help=argparse.SUPPRESS)
        # If dev_test_jaco is on, two new ct properties are added.
        # r_psp_MetadynamicsBinding_CompScore_test : CV/acj - 6.0*Hbond_persist
        # r_psp_MetadynamicsBinding_PoseScore_test : CV/acj
        # CV/acj :  averaging CV using simplified jacobian
        parser.add_argument(
            "-nposes",
            default=None,
            action=primeparser.store_int(min=1),
            help="Number of poses to re-score.\n"
            "By default all poses are scored but this can be used\n"
            "to score only the first N poses in the input file")
        parser.add_argument(
            "-selected_poses",
            nargs="+",
            default=None,
            help="The entries in the input file to run (or rerun).\n" +
            "This value should be 1-indexed.  Note that if you\n" +
            "are rerunning a job with the -previous_metadir flag\n" +
            "then this will be pose numbers in the output file of\n" +
            "the previous run.")
        parser.add_argument(
            "-ntrial",
            default=10,
            action=primeparser.store_int(min=1),
            help="Number of trials per pose (default is 10)")
        parser.add_argument(
            "-anchor_region",
            action=primeparser.StoreAsl,
            default=None,
            help="ASL expression of the anchor region.  This should be\n"
            "the backbone of 3 to 4 residues in what is thought to be\n"
            "the rigid core of the protein.  This should\n"
            "only be used in combination with 'ligand' cv")
        parser.add_argument(
            "-cv",
            choices=["ligand", self.DEFAULT_CV],
            default=self.DEFAULT_CV,
            help="Collective variable to use.  Choices are ligand &\n" +
            "{} (default).".format(self.DEFAULT_CV))
        parser.add_argument(
            "-height",
            action=primeparser.store_float(min=0.0),
            default=self.DEFAULT_HEIGHT,
            help="Bias potential height in the metadynamics simulation,\n" +
            "unit: kcal/mol, (default is %(default)s)")
        parser.add_argument(
            "-restrain",
            action=primeparser.StoreBoolean,
            default=True,
            help="Whether to use restraints (yes/no default is yes")
        parser.add_argument(
            "-smoothing",
            default=50,
            action=primeparser.store_int(min=0),
            help="Smoothing constant for CV (default:50)")
        parser.add_argument(
            "-score_fraction",
            default=self.DEFAULT_SCORE_FRACTION,
            action=primeparser.store_float(min=0.0, max=1.0),
            help="Set the fraction of the CV to use as the score for ranking.\n"
            "0.2 means the last 20 percent will be used.  (Default: %(default)s)"
        )
        parser.add_argument(
            "-label_property",
            default=self.DEFAULT_LABEL_PROPERTY,
            help="CT-level property to use as a label. Otherwise use\n"
            "the title of the structure")
        persistence_msg ="The -persistence argument has been removed.\n"+\
                     "All runs by default calculate the persistence score.\n"+\
                     "To view just the PoseScore use the\n"+\
                     "r_psp_MetadynamicsBinding_PoseScore ct-level property."
        parser.add_argument(
            "-persistence",
            action=primeparser.display_message_and_exit(persistence_msg),
            help=argparse.SUPPRESS)
        parser.add_argument(
            "-run_mode",
            choices=["all", "prep", "rescore", "md", "mtd"],
            default="all",
            help="Modify the mode of binding_pose_metadynamics.\n"
            "all     -- preps runs and scores the subjobs.\n"
            "prep    -- preps the jobs, but does not run them or score the\n"
            "           results.\n"
            "rescore -- for an already completed job, run or re-run the\n"
            "           scoring.\n"
            "md      -- Runs multisim up to and including the unbiased MD\n"
            "           stage. Does not perform the full workflow.\n"
            "           Does not score jobs.\n"
            "mtd     -- Runs the metadyanmics stage. Meant to be run on a\n"
            "           a previous job run with the 'md' run_mode.\n"
            "           Will also perform scoring. The jobname must match\n"
            "           the jobname used with the 'md' run.\n"
            "Note rescore MUST be run with the same settings as the original\n"
            "run.\n"
            "Default: %(default)s")
        parser.add_argument(
            "-previous_metadir",
            default=None,
            help="Add additional trials to an existing\n "
            "BindingPoseMetadynamics run.\n"
            "and with the same settings as the original run.  Note that\n"
            "the output maestro file should be used as the input")
        parser.add_argument(
            "-metadir_basename",
            default=None,
            help="Specify the name of the binding pose metadynamics workdir.\n"
            "If this option is not set, the jobname will be used to set the\n"
            "name of the workdir")
        parser.add_argument(
            "-report_md_equil_rmsd",
            action='store_true',
            help="For each trial, for each pose, will report the RMSD\n"
            "of the pose immediately after MD equlibration but\n"
            "before any metadynamic repulsion has been applied.\n"
            "This will also produce an additional output file,\n"
            "JOBNAME-mdequil_out.maegz, containing these poses.\n"
            "The RMSDs will also be printed to the logfile.")
        parser.add_argument(
            "-ligand_tail_for_md_equil_rmsd",
            action=primeparser.StoreSmarts,
            help="For -report_md_equil_rmsd ONLY, skip atoms defined by\n"
            "the given SMARTS patterns for the purposes of RMSD calculations.\n"
            "These atoms are considered ligand tails whose positions is not\n"
            "physically relevant")
        parser.add_argument(
            "-no_hmr",
            action='store_true',
            help="Do not use hydrogen mass repartitioning as part of MD\n"
            "Default is to use HMR")
        parser.add_argument(
            "-solvate",
            action=primeparser.StoreBoolean,
            default=False,
            help="Use solvate pocket as part of the multisim workflow\n"
            "(yes/no default is no)")
        parser.add_argument(
            "-remove_waters",
            action='store_true',
            # Hiding this option due to IFD-662.
            #help="Removes waters from the trajectories after the jobs " +
            #"have finished. This is purely meant to save disk space; " +
            #"results are not affected.")
            help=argparse.SUPPRESS)
        parser.add_argument(
            "-ignore_sh_hbonds",
            action='store_true',
            # Currently an option under scientific validation.
            # This ignores hbonds in which the donor hydrogen is
            # attached to a sulfur, for example, as in CYS.
            # The intent is to consider these hbonds too weak
            # to be of significance when considering hbond
            # persistence.
            help=argparse.SUPPRESS)
        # Debug argument that runs very short MD jobs to aid development
        # Should NOT be used for poduction runs
        parser.add_argument(
            "-dryrun", action="store_true", help=argparse.SUPPRESS)

        parser.add_argument(
            "-cloud",
            metavar='PRE_EMPTIBLE_CPU_HOST',
            help="The argument to -cloud is a pre_emptible CPU host used\n"
            "for the system builder jobs. -SUBHOST will be used\n"
            "exclusively for GPU subjobs and can be a pre-emptible\n"
            "GPU host")
        # This does a dryrun but uses the GPU
        parser.add_argument(
            "-dryrun_gpu", action="store_true", help=argparse.SUPPRESS)
        # Metadynamics stage production time in nano seconds.
        parser.add_argument(
            "-prod_time",
            default=self.DEFAULT_PROD_TIME,
            action=primeparser.store_float(min=0.0),
            help=argparse.SUPPRESS)

    def runBackend(self, opts, istage):
        """
        This is called by the executeparallel template to run the task
        opts {argparse.Namespace} options comming from the command-line
            parser
        istage {int} Not used
        """

        # For any OMP threading used by numpy, set the max threads to one
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(1)

        # Setup the metadynamics runs
        cts, jobs = self.setupMetadynamics(opts)
        # Set up a JobDJ.  If an MD host is provided then run the
        # metadynamics on a designated queue (for gpu jobs)
        kwargs = {
            'default_max_retries': opts.retries,
            'verbosity': 'verbose',
            'max_failures': 1
        }
        job_dj = queue.JobDJ(**kwargs)

        for one_ct_jobs in jobs:
            for job in one_ct_jobs:
                metadir_basename = self.metadir_basename(opts)
                # Check if the job is already completed
                add_job = self.checkIncompleteTrials(
                    job, self.get_job_dir(metadir_basename))
                if add_job and opts.previous_metadir:
                    add_job = self.checkIncompleteTrials(
                        job, opts.previous_metadir)
                if add_job:
                    job_dj.addJob(job)
        print("Prepped Jobs")
        if opts.run_mode == "prep":
            "Aborting before running jobs."
            return
        # Only run the jobs if requested
        if opts.run_mode in ['all', 'md', 'mtd']:
            # Due to JOBCON-5143, need to disable
            # smart distribution.
            # Otherwise, the head node will attempt
            # to launch a GPU job even though
            # the node itself may only have CPUs
            job_dj.disableSmartDistribution()
            if job_dj.total_added > 0:
                job_dj.run()
            else:
                print("No sub-jobs to add.")
        if opts.run_mode == 'md':
            print("MD-only multisim jobs completed.")
            # Nothing else to do once the multisim
            # jobs have completed.
            return
        if opts.remove_waters and not opts.run_mode == 'rescore':
            self.remove_waters(opts, jobs)
        self.getContactPersistence(opts, jobs, cts)
        self.rmsdAverageEstimator(opts, jobs, cts)

    def remove_waters(self, opts, jobs):
        for one_ct_jobs in jobs:
            for job in one_ct_jobs:
                job_tgz = self.getJobOutputFile(job)
                self.remove_single_tgz_waters(
                    job_tgz, self.INTERNAL_LIG_ASL, num_retain=0)

    @classmethod
    def remove_single_tgz_waters(cls, tgz_file, dew_asl, num_retain=0):
        """
           Runs trj_parch.py on the
           MD trajectory contained within {tgz_file}.
           The number of waters to retain (centered around
           the atoms selected by {dew_asl}) is specified by {num_retain}

           Afterwards, repacks the .tgz file but
           renames the ct title in the trajectory
           to contain the string '(no waters') if
           'num_retain' is 0, or '(local waters)'
           otherwise.

           The .tgz is effectively re-packed in place,
           removing the original .tgz file.
        """

        dir = os.path.dirname(tgz_file)
        tar = tarfile.open(tgz_file)
        enclosed_files = tar.getmembers()

        # Find the mtd stage dir
        top_level_dirs = []
        top_level_files = []
        for enclosed_file in enclosed_files:
            # If this file is within a subdirectory within the tar
            # Skip it as only looking for top-level files within the tar
            if os.path.dirname(enclosed_file.name):
                continue
            if enclosed_file.isdir():
                top_level_dirs.append(enclosed_file.name)
            else:
                top_level_files.append(enclosed_file.name)

        if len(top_level_dirs) > 1:
            error = "Tar extracts to more than one directory"
            raise RuntimeError(error)
        if not top_level_dirs:
            error = "Tar contains no sub-directories"
            raise RuntimeError(error)

        mtd_stage_dir = top_level_dirs[0]

        # Extract the tar in the dir containing the tar
        tar.extractall(path=dir)
        tar.close()

        cms_path = os.path.join(dir, mtd_stage_dir,
                                "{}-out.cms".format(mtd_stage_dir))
        trj_path = os.path.join(dir, mtd_stage_dir,
                                "{}_trj".format(mtd_stage_dir))

        # Create an indicator to reflect if no waters or only local waters are retained
        if num_retain == 0:
            retained_solvent = 'no_waters'
        else:
            retained_solvent = 'local_waters'

        out_path = os.path.join(dir, mtd_stage_dir, "{}_{}".format(
            mtd_stage_dir, retained_solvent))

        command = [
            'run', 'trj_parch.py', '-dew-asl', dew_asl, '-n',
            str(num_retain), '-align-asl', cls.TGZ_ALIGN_ASL, '-ref-asl',
            cls.TGZ_ALIGN_ASL, cms_path, trj_path, out_path
        ]
        print("Running trj_parch.py on {}".format(tgz_file))
        try:
            output = check_output(command, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print("trj_parch.py error:")
            print(e.output)
            raise

        less_water_temp_cms_path = out_path + '-out.cms'

        # Remove the old files
        shutil.rmtree(trj_path)
        os.remove(cms_path)
        os.remove(tgz_file)

        # Need to modify the cms file
        # to reflect the fact that there are fewer
        # waters now
        new_cms_path = out_path + '-new.cms'
        writer = structure.StructureWriter(new_cms_path)
        for ct in structure.StructureReader(less_water_temp_cms_path):
            formatted_retained_solvent = '({})'.format(
                retained_solvent.replace('_', ' '))
            ct.title = ct.title.replace('(full system)',
                                        formatted_retained_solvent)
            # Ensure that the path to the trajectory file refers
            # to the new name
            ct.property['s_chorus_trajectory_file'] = \
                    ct.property['s_chorus_trajectory_file'].replace('_{}'.format(retained_solvent), '')
            writer.append(ct)
        writer.close()

        # Rename the new files to replace the old files
        os.rename(new_cms_path, cms_path)
        os.rename(out_path + '_trj', trj_path)

        # And delete the cms file prior to my modifications
        os.remove(less_water_temp_cms_path)

        # Create the new tgz
        tar = tarfile.open(tgz_file, "w:gz")
        tar.add(os.path.join(dir, mtd_stage_dir), arcname=mtd_stage_dir)
        for top_level_file in top_level_files:
            tar.add(os.path.join(dir, top_level_file), arcname=top_level_file)
        tar.close()

        # Remove the unextracted directory
        shutil.rmtree(os.path.join(dir, mtd_stage_dir))
        for top_level_file in top_level_files:
            os.remove(os.path.join(dir, top_level_file))

    def preLaunchActions(self, opts, launch):
        """
        Before running the script, copy over the results of previous
        runs if requested.
        opts {argparse.Namspace} shared options
        launch {Launcher.Launcher} object that will be luanched via jobcontrol
           primarily passed so that we can add additional files that will
           be copied over to this object for jobcontrol processing.
        """
        # Copy over previous meta directories. This should be done regardless
        # of run_mode to increase restartability
        meta_dir = self.get_job_dir(self.metadir_basename(opts))

        if opts.previous_metadir:
            meta_dir = opts.previous_metadir

        if not os.path.isdir(meta_dir):
            # If the run mode is 'rescore' or 'mtd', then the metadynamics
            # directory must be present
            if opts.run_mode in ['rescore', 'mtd']:
                raise IOError(
                    "Cannot locate metadynamics results directory: " + meta_dir)
        else:
            for dir, subdirs, fns in os.walk(meta_dir):
                rel_dir = os.path.relpath(dir, os.getcwd())
                for fn in fns:
                    launch.addInputFile(os.path.join(rel_dir, fn))

    @classmethod
    def getJobOutputFile(cls, job, stage=None):
        """
        For a given job {queue.JobControlJob} and a stage number (stage, {int})
        within the multi_sim job created by job, return the name of the output
        tgz file as a string.
        """
        if stage is None:
            stage = cls.PRODUCTION_STAGE
        return os.path.join(job.getCommandDir(), "%s_%d-out.tgz" % (job.name,
                                                                    stage))

    def rmsdAverageEstimator(self, opts, jobs, cts):
        """
        Calculate the expectation value of the RMSD in the collective variable
        opts {argparse.Namspace} shared options
        jobs {list of {list of queue.JobControlJob}} the desmond muti_sim jobs
            correspoding to the run jobs.  This is essentially a matrix of
            JobControlJob object with the first axis corresponding to the pose
            and the second to the trial.
        cts {list of schrodinger.Structure} the input structure for each pose.
           This should be the same length as the first dimension of jobs
        Expectation values will be stored in the objects making up cts
        """
        pose_data = []
        pose_data_test = []  # store the corrected average rmsd
        for ict, one_ct_jobs in enumerate(jobs):
            trial_data = pd.DataFrame()
            trial_data_test = pd.DataFrame()
            for itrial, job in enumerate(one_ct_jobs):
                print("Processing Pose %d Trial %d" % (ict + 1, itrial + 1))
                # Extract the collective variable position
                tgz_fn = self.getJobOutputFile(job)
                output_dir = '{}_{}'.format(job.name, self.PRODUCTION_STAGE)
                of_fn = os.path.join(output_dir, "metadynamics_outfile.dat")
                of_path = os.path.join(job.getCommandDir(), of_fn)
                if not os.path.isfile(of_path):
                    tgz_fp = tarfile.open(tgz_fn, "r:gz")
                    tgz_fp.extract(of_fn, job.getCommandDir())
                chem_time, data, data_test = self.processOneOutFile(
                    opts, of_path)
                os.remove(os.path.join(job.getCommandDir(), of_fn))
                try:
                    os.rmdir(
                        os.path.join(job.getCommandDir(),
                                     os.path.dirname(of_fn)))
                except OSError:
                    # If the workdir resides on an NFS, the python process
                    # may create a .nfs* file which would prevent
                    # rmdir from completing. This excepts that error
                    # leaving the otherwise empty directory intact.
                    pass
                trial_data = trial_data.join(
                    pd.DataFrame(
                        {
                            'RMSDE%02d' % (itrial + 1): data
                        }, index=chem_time),
                    how="outer")
                trial_data_test = trial_data_test.join(
                    pd.DataFrame(
                        {
                            'RMSDE%02d' % (itrial + 1): data_test
                        },
                        index=chem_time),
                    how="outer")
            pose_data.append(trial_data)
            pose_data_test.append(trial_data_test)
        self.createOutput(opts, pose_data, cts, pose_data_test)

    def processOneOutFile(self, opts, fn):
        """
        For a given metadynamics cv file, get the data and expectation.
        opts {argparse.Namspace} shared options
        fn {string, path} Name of the desmond metadyanmics collective variable
            output file for the production metadyamics stage.
        Returns a tuple of arrays of the same length.  The first is the
            chemical time, the second is the expectation value of the RMSD
            calculate with all of the data up to that data point.
        """
        output = []
        output_test = []
        with open(fn, 'r') as fh:
            raw_temp = fh.readlines()
        # Remove lines where two lines are merged due to i/o problems
        raw_lines = [line for line in raw_temp if len(line.split()) == 4]
        raw_data = np.loadtxt(raw_lines)
        raw_data = np.array([d for d in raw_data if d[1] > 0.0000001])
        chem_time = raw_data[:, 0]
        height = raw_data[:, 1]
        center_0 = np.array(raw_data[:, 2])
        width_0 = np.array(raw_data[:, 3])
        # This correction here is to remove the few remaining situations
        # where the RMSD gets out of frame.
        # Also remove lines with high width due to potential i/o problems
        n_capped_gaussians = 0
        last_gaussian_center = center_0[0]
        last_gaussian_width = width_0[0]
        for i in range(len(center_0)):
            if not (center_0[i] <= self.MAX_GAUSSIAN_CENTER and
                    width_0[i] < self.MAX_GAUSSIAN_WIDTH):
                center_0[i] = last_gaussian_center
                width_0[i] = last_gaussian_width
                n_capped_gaussians += 1
            last_gaussian_center = center_0[i]
            last_gaussian_width = width_0[i]
        if n_capped_gaussians != 0:
            print("Capped the MtD gaussian at {} time steps".format(
                n_capped_gaussians))
        # get the range
        ptnl_min_0 = np.min(center_0 - width_0)
        ptnl_max_0 = np.max(center_0 + width_0)
        #Define the sample spacing for the bias potential table
        #This works ok for the rmsd cases--generalize/allow as input eventually
        n_samples = 100
        bin_width_0 = old_div((ptnl_max_0 - ptnl_min_0), float(n_samples))

        ptnl_table_0 = np.zeros(n_samples)
        for indx in range(len(height)):
            this_center = center_0[indx]
            the_height = height[indx]
            # Only do the integral over the range of values we have seen
            # up to this point.  This will avoid the superfluous peaks we
            # see at the begining of the trajectory.
            this_ptnl_min_0 = np.min(center_0[:indx + 1] - width_0[:indx + 1])
            this_ptnl_max_0 = np.max(center_0[:indx + 1] + width_0[:indx + 1])
            this_ptnl_imin = int(
                math.floor(
                    old_div(float(this_ptnl_min_0 - ptnl_min_0), bin_width_0)))
            this_ptnl_imax = n_samples - int(
                math.floor(
                    old_div(float(ptnl_max_0 - this_ptnl_max_0), bin_width_0)))
            R_samples = []
            #add the gaussian from this bias point to the bias potential array
            for a_bin in range(n_samples):
                dep_val = ptnl_min_0 + bin_width_0 * a_bin
                R_samples.append(dep_val)
                twidth = 2.0 * (width_0[0] * width_0[0])
                gaus_val = the_height * math.exp(-(dep_val - this_center) * (
                    dep_val - this_center) / twidth)
                ptnl_table_0[a_bin] += gaus_val
            #calculate rmsd average estimator with current bias potential
            # requires a value for beta
            expBV = []
            expBV_test = []
            RexpBV = []
            RexpBV_test = []
            kT = (0.0019872041 * 300)
            beta = old_div(1.0, kT)
            # This is equation 6 in doi: 10.1103/PhysRevLett.100.020603
            V_to_F_const = (kT + self.KTEMP) / self.KTEMP
            #build integrand sample arrays for numerator and denominator:
            for a_bin in range(this_ptnl_imin, this_ptnl_imax):
                RexpBV.append(R_samples[a_bin] * math.exp(
                    V_to_F_const * beta * ptnl_table_0[a_bin]))
                expBV.append(
                    math.exp(V_to_F_const * beta * ptnl_table_0[a_bin]))
                RexpBV_test.append(R_samples[a_bin]**3 * math.exp(
                    V_to_F_const * beta * ptnl_table_0[a_bin]))
                expBV_test.append(R_samples[a_bin]**2 * math.exp(
                    V_to_F_const * beta * ptnl_table_0[a_bin]))
            # just used numpy trapezoid-rule integrator for integration
            # for convenience
            if (indx != 0):
                numerator = np.trapz(
                    RexpBV,
                    R_samples[this_ptnl_imin:this_ptnl_imax],
                    dx=bin_width_0)
                denominator = np.trapz(
                    expBV,
                    R_samples[this_ptnl_imin:this_ptnl_imax],
                    dx=bin_width_0)
                ave_estimate1 = old_div(numerator, denominator)
                numerator_test = np.trapz(
                    RexpBV_test,
                    R_samples[this_ptnl_imin:this_ptnl_imax],
                    dx=bin_width_0)
                denominator_test = np.trapz(
                    expBV_test,
                    R_samples[this_ptnl_imin:this_ptnl_imax],
                    dx=bin_width_0)
                ave_estimate1_test = old_div(numerator_test, denominator_test)
            else:
                ave_estimate1 = this_center
                ave_estimate1_test = this_center
            output.append(ave_estimate1)
            output_test.append(ave_estimate1_test)
        return chem_time, output, output_test

    def createOutput(self, opts, pose_data, cts, pose_data_test):
        """
        Create the final output plots, output structures, and log file entries
        opts {argparse.Namspace} shared options
        pose_data {list of pandas.DataFrame} a list of data frames, one for
            echo conformation with the RMSD expectation results for each
            trial stored as columns and the chemical time as the index
        cts {list of schrodigner.Structure} should have the same size as
            pose_data.  Input conformations for each conformation.  They will
            be modified to include calcualted values.
        """
        # Create the plot
        error_bar_npoints = 5
        fig1 = plt.figure()
        plot11 = fig1.add_subplot(111)
        score_and_ct = []
        # Plotting error bars can still cause
        # clutter and should be left off
        # by default.
        plot_error_bars = False
        error_bar_offset = 0.1
        for ipose, one_pose_data in enumerate(pose_data):
            ct = cts[ipose]
            # If an original title property was set
            # then apply it now. If using a pre-built system,
            # this property is likely not present, hence the
            # try-except.
            try:
                ct.title = ct.property[self.ORIGINAL_TITLE_PROPERTY]
                # Clean up the property
                del ct.property[self.ORIGINAL_TITLE_PROPERTY]
            except KeyError:
                pass

            for saved_prop_name, actual_prop_name in self.GROUP_PROPERTY_PAIRS:
                if saved_prop_name in ct.property:
                    ct.property[actual_prop_name] = ct.property[saved_prop_name]
                    del ct.property[saved_prop_name]

            for atom in ct.atom:  #to remove temporaly added atom level property
                if self.ORIGINAL_ATOM_BOOL in atom.property:
                    del atom.property[self.ORIGINAL_ATOM_BOOL]

            # Score is the average score over the last part of the trajectory
            mean = np.array(one_pose_data.mean(1))
            mean_test = np.array(pose_data_test[ipose].mean(1))
            pose_score = np.mean(mean[int(
                math.floor(float(len(mean) * (1 - opts.score_fraction)))):])
            pose_score_test = np.mean(mean_test[int(
                math.floor(float(len(mean_test) * (
                    1 - opts.score_fraction)))):])
            ct.property['r_psp_MetadynamicsBinding_PoseScore'] = pose_score
            if opts.dev_test_jaco:
                ct.property[
                    'r_psp_MetadynamicsBinding_PoseScore_test'] = pose_score_test
            if ("r_psp_MetadynamicsBinding_HBond_Persistence" in ct.property):
                score = pose_score + \
                    -5.0 * ct.property["r_psp_MetadynamicsBinding_HBond_Persistence"]
                ct.property["r_psp_MetadynamicsBinding_CompScore"] = score
                if opts.dev_test_jaco:
                    score_test = pose_score_test + \
                        -6.0 * ct.property["r_psp_MetadynamicsBinding_HBond_Persistence"]
                    ct.property[
                        "r_psp_MetadynamicsBinding_CompScore_test"] = score_test
            else:
                score = pose_score
            score_and_ct.append((score, ct, one_pose_data))
        score_and_ct.sort()
        # Create the sorted output maegz file
        writer = structure.StructureWriter(opts.jobname + "-out.maegz")
        # Create the plot
        if ("r_psp_MetadynamicsBinding_CompScore" in ct.property):
            print("Composite Scores for each pose")
        else:
            print("CV Scores for each pose")
        print("%-40s , %s" % ("Pose", "Score"))

        raw_data_csv = opts.jobname + "-out.csv"
        csv_fh = open(
            raw_data_csv,
            'w',
            newline='',
        )
        csvwriter = csv.writer(csv_fh)
        time_points_in_ns = list(0.001 * score_and_ct[0][2].index)
        csv_header = ['Pose/Time (ns)'] + time_points_in_ns
        csvwriter.writerow(csv_header)

        for ipose, (score, ct, one_pose_data) in enumerate(score_and_ct):
            mean = np.array(one_pose_data.mean(1))
            rank = ipose + 1
            try:
                value = ct.property[opts.label_property]
                if opts.label_property.startswith("r_"):
                    label = "%2d:%0.3f" % (rank, value)
                else:
                    label = "%2d:%s" % (rank, value)
            except KeyError:
                label = "Pose %2d:%s" % (rank, ct.title)
            if ("r_psp_MetadynamicsBinding_HBond_Persistence" in ct.property):
                label += " PersScore %0.3f PoseScore %0.3f" % (
                    ct.property["r_psp_MetadynamicsBinding_HBond_Persistence"],
                    ct.property["r_psp_MetadynamicsBinding_PoseScore"])
            else:
                label += " Score %0.3f" % score
            if ("r_psp_MetadynamicsBinding_In_RMSD" in ct.property):
                label += " INRMSD %4.2f" % (
                    ct.property["r_psp_MetadynamicsBinding_In_RMSD"])
            step = int(math.floor(old_div(float(len(mean)), error_bar_npoints)))
            csv_row = [label] + list(mean)
            csvwriter.writerow(csv_row)
            # Keep track of the line plot as this will be included in the
            # legend, while others (ie the error bars) won't.
            plot11.plot(
                0.001 * one_pose_data.index,
                mean,
                self.COLORS[ipose % len(self.COLORS)],
                label=label)
            if plot_error_bars:
                # Plot a smaller number or error bar points so we don't clutter
                # the plot.
                step_offset = int(math.ceil(step * (1 - error_bar_offset)))
                for series_name in one_pose_data.columns:
                    plot11.scatter(
                        0.001 *
                        one_pose_data[series_name][step_offset::step].index,
                        one_pose_data[series_name][step_offset::step],
                        c=self.ERROR_BAR_COLORS[ipose % len(
                            self.ERROR_BAR_COLORS)],
                        s=20,
                        # Adding the _ removes it from legend
                        label="_ERROR BAR " + label)
            plot11.set_xlim([
                0.001 * one_pose_data.index[0], 0.001 * one_pose_data.index[-1]
            ])
            print("%-40s , %.3f" % (label, score))
            writer.append(ct)
        writer.close()
        csv_fh.close()
        print()
        plot11.set_ylabel('CV RMSD(A)')
        plot11.set_xlabel('MD Time (ns)')
        # Create the legend
        plot11.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=1,
            fancybox=True,
            shadow=True)
        # Output the plot
        plt.savefig(opts.jobname + "-out.pdf", bbox_inches='tight')
        plt.savefig(opts.jobname + "-out.png", bbox_inches='tight')
        print("Output plot stored as %s-out.png" % opts.jobname)
        print("Output plot stored as %s-out.pdf" % opts.jobname)

        plt.close(fig1)

    def getContactPersistence(self, opts, jobs, cts):
        """
        opts {argparse.Namespace} options comming from the command-line
            parser
        jobs { list of list of JobControlJob objects
            job) The top-level list contains one list for each pose and
            the second-level list contains one JobControlJob object
            for each trial.  The JobControlJob is the metadynamics jobs
            that has already been run for each pose-trial combination}
        """

        if opts.report_md_equil_rmsd:

            align_dir = 'md_equil_align_dir'
            if not os.path.exists(align_dir):
                os.mkdir(align_dir)
            md_equil_prealigned_mae = opts.jobname + '_md_equil_prealigned.maegz'
            md_equil_prealigned_mae = os.path.join(align_dir,
                                                   md_equil_prealigned_mae)
            md_equil_writer = structure.StructureWriter(md_equil_prealigned_mae)
            # Using the first pose as the alignment reference
            md_equil_writer.append(
                self.set_ligand_to_mol_1(self.INTERNAL_LIG_ASL, cts[0]))

        one_ct_persist_per_ipose = []
        for ipose, (ct, one_ct_jobs) in enumerate(zip(cts, jobs)):
            one_ct_persist = {}
            for itrial, job in enumerate(one_ct_jobs):
                tgz_fn = os.path.join(self.getJobOutputFile(job))
                tgz_fp = tarfile.open(tgz_fn, "r:gz")
                tgz_fp.extractall(job.getCommandDir())
                output_dir = '{}_{}'.format(job.name, self.PRODUCTION_STAGE)

                cmsout_fn = output_dir + '-out.cms'
                cmsout_fn = os.path.join(output_dir, cmsout_fn)
                cmsout_fn = os.path.join(job.getCommandDir(), cmsout_fn)

                cmstraj_fn = output_dir + '_trj'
                cmstraj_fn = os.path.join(output_dir, cmstraj_fn)
                cmstraj_fn = os.path.join(job.getCommandDir(), cmstraj_fn)

                one_traj_persist = self.getOneTrajPersistence(
                    opts, ct, cmsout_fn, cmstraj_fn)

                if opts.report_md_equil_rmsd:
                    self.get_md_equil_ct(self.INTERNAL_LIG_ASL, cmsout_fn,
                                         cmstraj_fn, ipose, itrial,
                                         md_equil_writer)

                self.cleanup_tarfile(tgz_fp, job.getCommandDir())
                tgz_fp.close()
                unused_cons = set(one_ct_persist)
                for con in one_traj_persist:
                    if (con not in one_ct_persist):
                        one_ct_persist[con] = [0] * (itrial)
                    else:
                        unused_cons.remove(con)
                    one_ct_persist[con].append(one_traj_persist[con])
                for con in unused_cons:
                    one_ct_persist[con].append(0)

            one_ct_persist_per_ipose.append(one_ct_persist)

        if opts.report_md_equil_rmsd:

            md_equil_writer.close()
            md_equil_rmsds = self.get_md_equil_rmsd(opts, cts, align_dir,
                                                    md_equil_prealigned_mae)

        for ipose, ct in enumerate(cts):
            print("--------------------------------")
            print("Persistence Scores for pose %2d " % (ipose + 1))

            one_ct_persist = one_ct_persist_per_ipose[ipose]
            for con in sorted(one_ct_persist):
                print("%14.3f (%s)  %s" %
                      (np.mean(one_ct_persist[con]),
                       " ".join(["%5.2f" % p for p in one_ct_persist[con]]),
                       con))
            if opts.report_md_equil_rmsd:
                md_equil_rmsds_str = '{:5.2f} ' * len(md_equil_rmsds[ipose])
                md_equil_rmsds_str = \
                        md_equil_rmsds_str.format(*md_equil_rmsds[ipose])
                md_equil_rmsds_str = md_equil_rmsds_str.rstrip()
                print("MD Equil RMSD: ({})".format(md_equil_rmsds_str))
            self.addPersistenceScore(ct, one_ct_persist, "")
            self.addPersistenceScore(ct, one_ct_persist, "_HBond", ["HBond"])
            self.addPersistenceScore(ct, one_ct_persist, "_PiPi", ["PiPi"])
            print("--------------------------------")

    def addPersistenceScore(self, ct, one_ct_persist, name, types=None):
        """
        Add a single persistence score for a single ct.
        ct {schrodinger.Structure} input conformation.  Persitence score
            will be added to this object as a ct-level property
        name {string} name of the property to add.  Will be added
            into the root name of r_psp_MetadynamicsBinding<name>_Persistence
        types {list of strings with either HBond or PiPi} List specifying the
            types of interactions to include.  Current only HBond and PiPi
            are supported.
        """
        pers = []
        by_trial_pers = None
        for con in sorted(one_ct_persist):
            if types is None or con.type in types:
                pers.append(np.mean(one_ct_persist[con]))
                if by_trial_pers is None:
                    by_trial_pers = [[] for i in one_ct_persist[con]]
                for itrial, trial_data in enumerate(one_ct_persist[con]):
                    by_trial_pers[itrial].append(trial_data)
        if by_trial_pers is not None:
            for itrial in range(len(by_trial_pers)):
                by_trial_pers[itrial] = np.mean(by_trial_pers[itrial])
        if len(pers) > 0:
            pers_score = np.mean(pers)
        else:
            pers_score = 0
        print("%s Mean %5.3f Number %2d Total %6.2f" % (name, pers_score,
                                                        len(pers),
                                                        np.sum(pers)))
        ct.property["r_psp_MetadynamicsBinding%s_Persistence" %name] = \
            pers_score
        ct.property["r_psp_MetadynamicsBinding%s_Persistence_Length" %name]=\
            len(pers)
        ct.property["r_psp_MetadynamicsBinding%s_Persistence_Sum" %name] = \
            np.sum(pers)
        if by_trial_pers is not None:
            for itrial, score in enumerate(by_trial_pers):
                ct.property["r_psp_MetadynamicsBinding%s_Persistence_Trial_%d" %
                            (name, itrial + 1)] = score

    def getOneTrajPersistence(self, opts, input_ct, cmsout_fn, cmstraj_fn):
        """
        Calcuate the persistance score for a single trajectory of a single
        conformation.
        opts {argparse.Namespace} options coming from the command-line
            parser
        cmsoutfn & cmstraj_fn {string, path} Name of the output trajectory
            coming from desmond to process.  This is in cms format so there
            should be two files: an structure file and a trajectory file
           corresponding to cmsoufn and cmstrajfn respectively
        """
        # Prep the frames
        input_lig_ct = input_ct.extract(
            analyze.evaluate_asl(input_ct, self.INTERNAL_LIG_ASL),
            copy_props=True)
        input_rec_ct = input_ct.copy()
        input_rec_ct.deleteAtoms(
            analyze.evaluate_asl(input_rec_ct, "(%s)" % self.INTERNAL_LIG_ASL))
        _, cms_model = topo.read_cms(cmsout_fn)
        trj = traj.read_traj(cmstraj_fn)

        contacts = {}
        lic_kwargs = {
            'match_type': 'i_psp_MetadynamicsBinding_Index',
            'use_symmetry': True,
            'ignore_sh_hbonds': opts.ignore_sh_hbonds,
            'include_pipi': False
        }
        lic = ps.LigandInteractionComparison(input_lig_ct, input_rec_ct,
                                             **lic_kwargs)
        lic.hbond_dist_max = 2.8
        lic.hbond_dist_dev = 0
        lic.hbond_donor_angle_min = 120
        lic.hbond_donor_angle_dev = 0
        lic.hbond_acceptor_angle_min = 90
        lic.hbond_acceptor_angle_dev = 0
        LAST_N_FRAMES = 11
        for frame in trj[-LAST_N_FRAMES:]:
            # Use the LAST_N_FRAMES, with default input params. it's approx ~2ns
            frame_ct = topo.update_ct(cms_model.fsys_ct, cms_model, frame)
            frame_lig_ct = frame_ct.extract(
                analyze.evaluate_asl(frame_ct, self.INTERNAL_LIG_ASL),
                copy_props=True)
            frame_rec_ct = frame_ct.copy()
            frame_rec_ct.deleteAtoms(
                analyze.evaluate_asl(
                    frame_rec_ct,
                    "res.ptype SPC or (%s)" % self.INTERNAL_LIG_ASL))
            shared, unique = lic.getUniqueContacts(frame_lig_ct, frame_rec_ct)
            for con in itertools.chain(shared, unique):
                if con not in contacts:
                    contacts[con] = 0
            for con in shared:
                contacts[con] += 1
        for con in sorted(contacts):
            one_con_pers = old_div(
                float(contacts[con]), len(trj[-LAST_N_FRAMES:]))
            contacts[con] = one_con_pers
        return contacts

    def readCollectiveVariables(self, opts, jobs, cts):
        """
        opts {argparse.Namespace} options comming from the command-line
            parser
        jobs { list of list of JobControlJob objects
            job) The top-level list contains one list for each pose and
            the second-level list contains one JobControlJob object
            for each trial.  The JobControlJob is the metadynamics jobs
            that has already been run for each pose-trial combination}
        """
        # Get the time and the Collective variable out of the
        # cvseq file inside the gzipped tarfile
        cv_pattern = re.compile(r"\[cv_00\s+\[([\d\.\-]+)\]\]")
        time_pattern = re.compile(r"\[time\s+\[([\d\.\-]+)\]\]")
        cvs = []
        times = []
        for one_ct_jobs in jobs:
            one_ct_cvs = []
            for job in one_ct_jobs:
                # Extract the collective variable position
                tgz_fn = self.getJobOutputFile(job)
                output_dir = '{}_{}'.format(job.name, self.PRODUCTION_STAGE)
                cvseq_fn = output_dir + '.cvseq'
                cvseq_fn = os.path.join(output_dir, cvseq_fn)
                cvseq_path = os.path.join(job.getCommandDir(), cvseq_fn)
                if not os.path.isfile(cvseq_path):
                    tgz_fp = tarfile.open(tgz_fn, "r:gz")
                    tgz_fp.extract(cvseq_fn, job.getCommandDir())
                cv_list = []
                time_list = []
                for line in open(cvseq_path, "r").readlines():
                    match = cv_pattern.search(line)
                    time_match = time_pattern.search(line)
                    if match and time_match:
                        cv_list.append(eval(match.group(1)))
                        time_list.append(eval(time_match.group(1)))
                one_ct_cvs.append(cv_list)
            cvs.append(one_ct_cvs)
            times.append(time_list)
        times = np.array(times)  # The first dimension is the the pose,
        # second is the timestep
        cvs = np.array(cvs)  # First dimension is the pose, second is the trial,
        # Third is the timestep
        return times, cvs

    @staticmethod
    def markTerminusDistance(ct):
        """
        Write the sequence-space distance from the C and N Terminus to the ct
        in the atom-level property i_psp_nterm_res_number and
        i_psp_cterm_res_number where 1 means we are at the terminus (the
        units are residues)
        """
        for mol in ct.molecule:
            mol_res = [r for r in structure.get_residues_by_connectivity(mol)]
            for ires, res in enumerate(mol_res):
                nterm_ires = ires + 1
                cterm_ires = len(mol_res) - ires
                for atom in res.atom:
                    atom.property["i_psp_nterm_res_number"] = nterm_ires
                    atom.property["i_psp_cterm_res_number"] = cterm_ires

    def markBindingSite(self,
                        opts,
                        cts,
                        sys_build_lig_asl,
                        binding_site_radius=7.0,
                        terminus_cutoff=10):
        """
        Mark the atoms of the ligand with the asl expression
        atom.r_mtd_rmsd_weight = 1.0 and the binding site CAs with
        atom.r_mtd_superpos_weight = 1.0.  The binding site is defined as
        all residues within the <binding_site_radius> of the any ligand atom
        in any pose that are also not within <terminus cutoff> of the
        C or N terminus.
        opts {argparse.Namespace} options comming from the command-line
            parser
        cts { list of schrodinger.Structure} Structure to mark up, uderlying
           objects in the list will be modified
        binding_site_radius {integer} distance in A from the ligand that
           defines the binding site
        terminus_cutoff { integer} residue distance from the terminus of
           the protein to allow for binding site definition
        """
        # Mark the ligand
        for ct in cts:
            for atom in ct.atom:
                if "r_mtd_rmsd_weight" in atom.property:
                    del atom.property["r_mtd_rmsd_weight"]
                if "r_mtd_superpos_weight" in atom.property:
                    del atom.property["r_mtd_superpos_weight"]
            lig_iatoms = analyze.evaluate_asl(
                ct, "%s and not atom.ele H" % sys_build_lig_asl)
            if len(lig_iatoms) == 0:
                raise RuntimeError("No atoms match %s in %s" %
                                   (sys_build_lig_asl, ct.title))
            for iatom in lig_iatoms:
                ct.atom[iatom].property["r_mtd_rmsd_weight"] = 1.0
                ct.atom[iatom].property[self.INTERNAL_LIG_PROPERTY] = True
                ct.atom[iatom].property[self.GCMC_LIG_PROPERTY] = 1
        # Do the alignments
        if len(cts) > 1:
            marked_cts = structurealign.align_binding_sites(
                opts.jobname + ".align_binding_sites",
                cts,
                self.INTERNAL_LIG_ASL,
                in_place=True,
                radius=binding_site_radius)
        elif len(cts) == 1:
            temp_marked_cts = structurealign.align_binding_sites(
                opts.jobname + ".align_binding_sites", [cts[0], cts[0].copy()],
                self.INTERNAL_LIG_ASL,
                in_place=True,
                radius=binding_site_radius)
            marked_cts = temp_marked_cts[:1]
        self.markActiveSiteAtoms(marked_cts, terminus_cutoff, 10)

    @classmethod
    def getResNamesToUse(cls, cts, terminus_cutoff, minimum_nres):
        """
        cts {list of structure.Structure } Structures to process.  One
           atom on each residue in the binding site (usually the alpha
           carbon for protein or C5' for NA) should be marked with
           a unique tag in the SKA_RES_NAME atom property
        terminus_cutoff { integer } all residues within this number
            from the terminus should be skipped
        minimum_residues { integer } keep relaxing the terminus cutoff
            until this number of residues are found
        """
        # Create dictionaries with the distance from the N and C terminus
        # of each residue name
        distance_from_term = {}
        cts_present = {}
        for ict, ct in enumerate(cts):
            for mol in ct.molecule:
                mol_res = [
                    r for r in structure.get_residues_by_connectivity(mol)
                ]
                for ires, res in enumerate(mol_res):
                    nterm_ires = ires + 1
                    cterm_ires = len(mol_res) - ires
                    for atom in res.atom:
                        if cls.SKA_RES_NAME in atom.property:
                            rname = atom.property[cls.SKA_RES_NAME]
                            if rname in distance_from_term:
                                distance_from_term[rname] = min(
                                    distance_from_term[rname], nterm_ires,
                                    cterm_ires)
                            else:
                                distance_from_term[rname] = min(
                                    nterm_ires, cterm_ires)
                            if rname not in cts_present:
                                cts_present[rname] = set()
                            cts_present[rname].add(ict)
        rnames_to_use = None
        this_cutoff = terminus_cutoff
        while (rnames_to_use is None or len(rnames_to_use) < minimum_nres):
            rnames_to_use = [
                rname for rname in distance_from_term.keys()
                if (distance_from_term[rname] > this_cutoff and
                    len(cts_present[rname]) == len(cts))
            ]
            this_cutoff -= 1
            if this_cutoff == 0:
                break
        return rnames_to_use

    @classmethod
    def markActiveSiteAtoms(cls, marked_cts, terminus_cutoff, minimum_residues):
        """
        Mark the active site atoms given the data provided by align
        binding sites
        marked_cts {list of structure.Structure } Structures to process.  One
           atom on each residue in the binding site (usually the alpha
           carbon for protein or C5' for NA) should be marked with
           a unique tag in the SKA_RES_NAME atom property
        terminus_cutoff { integer } all residues within this number
            from the terminus should be skipped
        minimum_residues { integer } keep relaxing the terminus cutoff
            until this number of residues are found
        """
        atoms_by_residue_temp = {}
        for ict, ct in enumerate(marked_cts):
            for atom in ct.atom:
                if cls.SKA_RES_NAME in atom.property:
                    rname = atom.property[cls.SKA_RES_NAME]
                    if rname not in atoms_by_residue_temp:
                        atoms_by_residue_temp[rname] = [None] * len(marked_cts)
                    atoms_by_residue_temp[rname][ict] = atom
        atoms_by_residue = {
            key: value
            for key, value in atoms_by_residue_temp.items()
            if not (None in value)
        }
        res_names = BindingPoseMetadynamics.getResNamesToUse(
            marked_cts, terminus_cutoff, minimum_residues)
        print("Marked active site residues: %s" % (";".join(atoms_by_residue)))
        # Select out the atoms to mark
        any_selected = False
        for rname in res_names:
            include = True
            if None in atoms_by_residue[rname]:
                raise RuntimeError("Residue %s not found in input structure" %
                                   (rname))
            for ict, atom in enumerate(atoms_by_residue[rname]):
                for res_atom in atom.getResidue().atom:
                    if res_atom.pdbname in [
                            " C  ", " N  ", " O  ", " CA ", " P  ", " O5'",
                            " C5'", " C4'", " C3'", " O3'"
                    ]:
                        res_atom.property["r_mtd_superpos_weight"] = 1.0
                        any_selected = True
        if not any_selected:
            raise RuntimeError(
                "Could not identify backbone atoms " +
                "around the ligand.  Please check that they use " +
                'standard pdb naming: " C  ", " N  ", " O  ", " CA " ' +
                'for proteins and " P  ", " O5\'", " C5\'", " C4\'", ' +
                '" C3\'", "O3\'" for nuecleic acids.')

    def getPoseInformation(self, ct):
        """Little one liner that exists solely to allow for easier mocking"""
        return ct.property.get(self.POSE_TAG)

    def createSystems(self, opts):
        """
        Return the structures, with trial information set up, as a list
        of schrodinger.Structure objects
        opts {argparse.Namespace} shared options
        fn {string, path} name of the file with input structures
        """
        # Get the selected number of poses
        pose_numbers = self.getSelectedPoseNumbers(opts)
        ipose = 0
        cms_fns = []
        jobs = []
        for fn in opts.struct_file:
            if fn.endswith(".cms"):
                if ipose not in pose_numbers:
                    continue
                msys_model, cms_model = topo.read_cms(self.getPath(fn))
                if opts.previous_metadir:
                    if (self.getPoseInformation(cms_model.fsys_ct) is None):
                        raise RuntimeError(
                            "%s " %
                            (fn) + "does not contain trial information.  " +
                            "Is the input file the output of a previous run?")
                    this_ipose = None
                else:
                    this_ipose = ipose
                if this_ipose is not None:
                    cms_model.property[self.POSE_TAG] = this_ipose
                job_dir = self.getRunDir(opts, ipose)
                pose_id = Stage.pose_id(ipose + 1, abbrev=True)
                out_cms_fn = os.path.join(job_dir,
                                          "{}_SystemBuilder_{}-out.cms".format(
                                              opts.jobname, pose_id))
                self.addOutputFile(out_cms_fn)
                # For an input cms file, the system is already solvated so there is
                # no certain way to acertain which atoms to retain. Therefore, we
                # keep all input atoms in the binding pose metadynamics output.
                # Adding a unit test for this as part of PRIME-4461
                for ct in cms_model.comp_ct:
                    for atom in ct.atom:
                        atom.property[self.ORIGINAL_ATOM_BOOL] = True

                cms_model.synchronize_fsys_ct()
                cms_model.write(out_cms_fn)
                cms_fns.append(os.path.abspath(out_cms_fn))
                # No need to update the ASL as the system has already been
                # built
                create_systems_ligand_asl = opts.ligand_asl
                ipose += 1
            else:
                for ict, ct in enumerate(
                        structure.StructureReader(self.getPath(fn))):
                    if ipose not in pose_numbers:
                        continue
                    set_ligand_iatoms = set(
                        analyze.evaluate_asl(ct, opts.ligand_asl))
                    if len(set_ligand_iatoms) == 0:
                        raise RuntimeError(
                            "%s(%d)%s " % (fn, ict + 1, ct.title) +
                            "does not contain any atoms that match the " +
                            "ligand asl expression %s" % opts.ligand_asl)
                    # Mark the appropriate atoms as in the ligand here
                    # to avoid issues down the roadi if the asl no longer
                    # applies after building the system
                    for atom in ct.atom:
                        atom.property[
                            self.
                            ORIGINAL_ATOM_BOOL] = True  #for tracking the original atoms in the input ct
                        if int(atom) in set_ligand_iatoms:
                            atom.property[self.INTERNAL_LIG_PROPERTY] = True
                            atom.property[self.GCMC_LIG_PROPERTY] = 1
                        else:
                            if self.INTERNAL_LIG_PROPERTY in atom.property:
                                del atom.property[self.INTERNAL_LIG_PROPERTY]
                            if self.GCMC_LIG_PROPERTY in atom.property:
                                del atom.property[self.GCMC_LIG_PROPERTY]
                    create_systems_ligand_asl = \
                        '(atom.{})'.format(self.INTERNAL_LIG_PROPERTY)
                    job_dir = self.getRunDir(opts, ipose)
                    ct.property[self.POSE_TAG] = ipose

                    # Backing up the original title and group properties
                    ct.property[self.ORIGINAL_TITLE_PROPERTY] = ct.title
                    for saved_prop_name, actual_prop_name in self.GROUP_PROPERTY_PAIRS:
                        if actual_prop_name in ct.property:
                            ct.property[saved_prop_name] = ct.property[
                                actual_prop_name]
                        else:
                            ct.property[saved_prop_name] = ""

                    if opts.previous_metadir:
                        if self.getPoseInformation(ct) is None:
                            raise RuntimeError(
                                "%s(%d)%s " % (fn, ict, ct.title) +
                                "does not contain trial information.  " +
                                "Is the input file the output of a previous run?"
                            )

                    job = CreateSystemJob(
                        ipose + 1, ct, opts.no_hmr, command_dir=job_dir)
                    cms_fns.append(os.path.abspath(job.out_cms_path))
                    if not os.path.exists(cms_fns[-1]):
                        jobs.append(job)
                    else:
                        local_fn = os.path.relpath(cms_fns[-1], job_dir)
                        print(
                            "%s exists from the %s directory and was skipped" %
                            (local_fn, job_dir))
                    for one_fn in job.files_to_save(ipose + 1, job_dir):
                        self.addOutputFile(one_fn)
                    ipose += 1
        return cms_fns, jobs, create_systems_ligand_asl

    def getSelectedPoseNumbers(self, opts):
        """
        Return indexes of selected poses
        """
        nct = 0
        for fn in opts.struct_file:
            if fn.endswith(".cms"):
                nct += 1
            else:
                nct += structure.count_structures(self.getPath(fn))
        if opts.selected_poses:
            poses = [int(p) - 1 for p in opts.selected_poses]
            if any(np.array(poses) >= nct):
                raise RuntimeError(
                    ("Some selected poses (%s) are "
                     "beyond the number of structures in the input files") %
                    (",".join([str(p + 1) for p in poses])))
        elif opts.nposes:
            poses = [p for p in range(min(opts.nposes, nct))]
        else:
            poses = [p for p in range(nct)]
        if len(set(poses)) != len(poses):
            raise RuntimeError("Seleted poses cannot contain duplicates: " +
                               " ".join([str(p + 1) for p in poses]))
        return poses

    def getCtsFromCMSFiles(self, cms_fns):
        """
        Generator that yields the schrodinger.Structure objects of the systems
        passed into it with all of the solvent removed
        cms_fns { iterator of strings } Paths to the cms files.  One structure
            object will be yielded for each filename
        """
        for cms_fn in cms_fns:
            msys_model, cms_model = topo.read_cms(cms_fn)
            # Even though we don't save this, it will be the same when
            # we load it in again
            self.setOriginalIndexProperty(cms_model)
            ct = cms_model.fsys_ct
            out_ct = ct.extract(
                analyze.evaluate_asl(ct, '(atom.{})'.format(
                    self.ORIGINAL_ATOM_BOOL)),
                copy_props=True)  # extract the original atoms
            yield out_ct

    @staticmethod
    def setOriginalIndexProperty(cms_model):
        """
        Add an atom-level property to both the full system ct and the
        component cts in a cms model so that they can be referenced later.
        This will primarily be used in the full system ct, but since the
        full system ct is often rebuilt from the component cts, it is added
        there as well.
        cms_model { cms.Cms } Cms model to modify
        """
        for iatom, (fsys_atom, comp_atom, comp_ct, comp_index) in \
                enumerate(topo.cms_atom(cms_model)):
            fsys_atom.property["i_psp_original_index"] = iatom
            comp_atom.property["i_psp_original_index"] = iatom

    @staticmethod
    def getRunDir(opts, ipose):

        metadir_basename = BindingPoseMetadynamics.metadir_basename(opts)
        metadir = BindingPoseMetadynamics.get_job_dir(metadir_basename)
        dir = os.path.join(metadir, Stage.pose_id(ipose + 1))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        return dir

    @classmethod
    def checkIncompleteTrials(cls, multisim_job, metadir):
        """
        Check to see if specific BP MtD trials are already missing/incomplete or
        if they have been completed previously

        metadir {str} path to the metadynamics directory where the jobs
            should be examined for previous completion
        multisim_job {JobControlJob object} A JobControlJob object used to specify the
            BP MtD job for each trial of each pose. The job is set to run in
            the trial-level directory

        return {bool} True if the output tgz file does not currently exist,
            False if the output tgz file has already been created, indicating
            the job for this trial was already completed
        """
        out_tgz_missing = True

        #get the jobname from metadir
        original_jobname = metadir[:-(len(cls.METADIR_POSTFIX) + 1)]
        #add '_' to match with the max length of jobname (6) using get_multisim_job_base
        bp_mtd_jobname = cls.get_multisim_job_base(original_jobname)

        trial_dir = os.path.join(*(
            [metadir] + split_path(multisim_job.getCommandDir())[1:]))
        prev_name = bp_mtd_jobname[:cls.
                                   MAX_JOBNAME_BASE_LEN] + multisim_job.name[cls.
                                                                             MAX_JOBNAME_BASE_LEN:]
        output_tgz = '{}_{}-out.tgz'.format(prev_name, cls.PRODUCTION_STAGE)
        existing_tgz = os.path.join(trial_dir, output_tgz)
        if os.path.isfile(existing_tgz):
            print("%s is already complete" % multisim_job.name)
            out_tgz_missing = False

        return out_tgz_missing

    def setupMetadynamics(self, opts):
        """
        opts {argparse.Namespace} options comming from the command-line
            parser
        The output will be a list of lists of JobControlJob objects.  The
            top-level list contains one list for each pose and the second-level
            list contains one object for each trial.  Each JobControlJob
            object is the metadynamics job to run for that pose-trial
            combintation
        """
        # Create the systems to process
        sys_build_cms_fns, sys_build_jobs, sys_build_lig_asl = \
            self.createSystems(opts)
        job_dj = queue.JobDJ(
            verbosity="verbose", default_max_retries=opts.retries)
        [job_dj.addJob(job) for job in sys_build_jobs]

        if opts.cloud:
            gpu_subhost, n_cpu = list(job_dj._hosts.items())[0]
            job_dj.setHostList([(opts.cloud, n_cpu)])

        if len(sys_build_jobs) > 0:
            print("Running system builder jobs")
            job_dj.run()
        cts = list(self.getCtsFromCMSFiles(sys_build_cms_fns))

        orig_dir = os.getcwd()
        jobs = []
        if opts.cv == "ligand_in_site" or opts.restrain:
            self.markBindingSite(opts, cts, sys_build_lig_asl)
        lig_smarts = None
        for ict, (sys_build_cms_fn, ct) in enumerate(
                zip(sys_build_cms_fns, cts)):
            # Generate a warning if there are different ligands
            lig_iatoms = analyze.evaluate_asl(ct, self.INTERNAL_LIG_ASL)
            if len(lig_iatoms) == 0:
                raise RuntimeError("Could not find ligand asl %s in %s(%d)" %
                                   (self.INTERNAL_LIG_ASL, ct.title, ict + 1))
            # Mark the ligand atoms
            # Make sure that any existing ligand properties are overwritten
            set_lig_iatoms = set(lig_iatoms)
            for atom in ct.atom:
                if int(atom) in set_lig_iatoms:
                    atom.property[self.INTERNAL_LIG_PROPERTY] = True
                    atom.property[self.GCMC_LIG_PROPERTY] = 1
                elif self.INTERNAL_LIG_PROPERTY in atom.property:
                    del atom.property[self.INTERNAL_LIG_PROPERTY]
                    del atom.property[self.GCMC_LIG_PROPERTY]
            lig_ct = ct.extract(lig_iatoms)
            if lig_smarts is None:
                lig_smarts = analyze.generate_smarts_canvas(lig_ct)
            else:
                if (len(analyze.evaluate_smarts_canvas(lig_ct,
                                                       lig_smarts)) == 0):
                    print("WARNING: Ligand in %s(%d) " % (ct.title, ict + 1) +\
                        "has a different chemical structure")

            # Mark the atoms with indexes to use with Persistence scoring later
            for atom in ct.atom:
                atom.property["i_psp_MetadynamicsBinding_Index"] = int(atom)
            ipose = ct.property[self.POSE_TAG]
            if opts.cv == "ligand_in_site" and opts.anchor_region is not None:
                raise RuntimeError("ligand_in_site CV not compatible with " +
                                   "an anchor region")
            elif opts.anchor_region is not None:
                cv_asl = "(%s) OR (%s)" % (self.INTERNAL_LIG_ASL,
                                           opts.anchor_region)
                restrain_info = ""
            elif opts.cv == "ligand_in_site":
                cv_asl = "(atom.r_mtd_rmsd_weight > 0) OR "+\
                         "(atom.r_mtd_superpos_weight >0)"
            else:
                cv_asl = self.INTERNAL_LIG_ASL
            if opts.restrain:
                restrain_info = \
                    'restrain = { '+\
                        'atom = "asl:atom.r_mtd_superpos_weight > 0" '+\
                        'force_constant = 0.05 '+\
                        'reference_position = reset }'
                #        'sigma = 2.5 '+\
            else:
                restrain_info = ""

            dir_name = self.getRunDir(opts, ipose)
            one_ct_jobs = []
            if not opts.run_mode == 'rescore':
                self.updateCmsFileAtomProperties(sys_build_cms_fn, ct,
                                                 sys_build_cms_fn)
            for itrial in range(opts.ntrial):
                meta_dir = self.metadir_basename(opts)
                trial_dir = self.get_trial_dir(
                    self.metadir_basename(opts), ipose, itrial)
                if (not os.path.isdir(trial_dir)):
                    os.makedirs(trial_dir)
                try:
                    os.chdir(trial_dir)
                    # The seeds should be different, but otherwise there
                    # is no reason they should be special
                    self.createMetadynamicsMSJ(opts, self.MTD_MSJ_FILENAME,
                                               cv_asl, restrain_info,
                                               1000 + itrial)
                    multisim_jobname = self.get_multisim_jobname(
                        opts.jobname, ipose, itrial)
                    cmd = [
                        os.path.join(
                            os.getenv('SCHRODINGER'), 'utilities', 'multisim'),
                        '-JOBNAME',
                        multisim_jobname,
                        '-mode',
                        'umbrella',
                        '-m',
                        self.MTD_MSJ_FILENAME,
                        os.path.relpath(sys_build_cms_fn),
                        '-cpu',
                        '1',
                        '-lic',
                        "DESMOND_GPGPU:16",
                    ]

                    if opts.run_mode == 'mtd':
                        # Need to continue the previous job
                        checkpoint_file = multisim_jobname + '-multisim_checkpoint'
                        # We must go back two stages from the production
                        # stage. One stage back is the 'stop' stage
                        tgz_file = '{}_{}-out.tgz'
                        tgz_file = tgz_file.format(multisim_jobname,
                                                   self.PRODUCTION_STAGE - 2)
                        # We restart going back one stage
                        # to the 'stop' stage which when re-run
                        # will be skipped.
                        checkpoint_file_with_stage = '{}:{}'.format(
                            checkpoint_file, self.PRODUCTION_STAGE - 1)
                        cmd.extend([
                            '-RESTART', checkpoint_file_with_stage, '-d',
                            tgz_file
                        ])
                    if not opts.dryrun:
                        cmd.extend([
                            '-set', 'stage[1].set_family.md.jlaunch_opt=[-gpu]'
                        ])
                    job = queue.JobControlJob(
                        cmd, command_dir=trial_dir, name=multisim_jobname)
                    one_ct_jobs.append(job)

                    files_to_save = \
                            self.get_multisim_files_to_save(multisim_jobname,
                                                            opts.run_mode)

                    for fn in files_to_save:
                        self.addOutputFile(
                            os.path.join(job.getCommandDir(), fn))
                finally:
                    os.chdir(orig_dir)
            jobs.append(one_ct_jobs)
        return cts, jobs

    def updateCmsFileAtomProperties(self, initial_cms_fn, ct, output_cms_fn):
        """
        Given an input cms model and a structure that has atom properties
        added to it, transfer those atomm preoperties from the ct to
        the cms model and output that to a given filename.  This assumes
        that the ct was pulled from either the full system or one of the
        component cts that make up the cms model in initial_cms_fn and was
        run through setOriginalIndexProperty before being modified to remove i
        atoms or add atom-level properties
        inputs
        inital_cms_fn { string, path} path to the cms_model to start with
        ct { schrodinger.Structure} structure to get atom-level properties from
            it is assumed that this is a subset of the structure in
            inital_cms_fn and that the cms_model used to generate it when
            through setOriginalIndexProperty beforehand
        output_cms_fn { string, path} output path to write the cms_model
            with atom properties added
        """
        msys_model, cms_model = topo.read_cms(initial_cms_fn)
        # As we are using the same files as before the indexes will be
        # unchanged from the ct
        self.setOriginalIndexProperty(cms_model)
        cms_original_index_to_ct_atom = {}
        for atom in ct.atom:
            cms_original_index_to_ct_atom[atom.property[
                "i_psp_original_index"]] = atom
        for iatom, (fsys_atom, comp_atom, comp_ct, comp_index) in \
                enumerate(topo.cms_atom(cms_model)):
            if "i_psp_original_index" in fsys_atom.property:
                psp_index = fsys_atom.property["i_psp_original_index"]
            elif "i_psp_original_index" in comp_atom.property:
                psp_index = comp_atom.property["i_psp_original_index"]
            else:
                continue
            ct_atom = cms_original_index_to_ct_atom.get(psp_index)
            if ct_atom is None:
                continue
            for prop in ct_atom.property:
                prop_keys = prop.split("_", 2)
                if (prop_keys[1] not in ["psp", "mtd"] or
                        prop == "i_psp_original_index"):
                    continue
                comp_atom.property[prop] = ct_atom.property[prop]
        # This will copy the atom information in the component ct's back
        # to the full system cts
        cms_model.synchronize_fsys_ct()
        cms_model.write(output_cms_fn)

    def createMetadynamicsMSJ(self, opts, fn, cv_asl, restrain_info, seed):
        """
        Write a file containing the metadynamics msj file for use with
        multisim.
        opts {argparse.Namespace} options comming from the command-line
            parser
        fn {string} Path of file to write (should end in msj)
        cv_asl {string of ASL expression} ASL pattern to use for
           collective variable in metadynamics
        restrain_info {string of multisim command} Multisim format
           specifying the atoms to constrain the the potential to use
        seed {integer} Random number seed to use

        """
        values = {
            'seed1': seed,
            'cv_asl': cv_asl,
            'restrain_info': restrain_info,
            'height': opts.height,
            'ktemp': self.KTEMP,
        }
        if opts.dryrun or opts.dryrun_gpu:
            values['equib1_time'] = 2
            values['equib2_time'] = 2
            values['equib3_time'] = 2
            values['equib4_time'] = 2
            values['equib5_time'] = 2
            values['equib6_time'] = 2
            values['equib7_time'] = 2
            values['prod_time'] = 5
        else:
            values['equib1_time'] = 100
            values['equib2_time'] = 12
            values['equib3_time'] = 12
            values['equib4_time'] = 500
            values['equib5_time'] = 500
            values['equib6_time'] = 12
            values['equib7_time'] = 500
            values['prod_time'] = opts.prod_time * 1000

        if opts.no_hmr:
            values['hmr_timestep'] = ''
        else:  # If we are using HMR (default)
            values['hmr_timestep'] = 'timestep = [0.004 0.004 0.008]'

        if opts.solvate:
            values['solvate_pocket'] = ''
        else:
            values['solvate_pocket'] = 'should_skip = true'

        if opts.run_mode == 'md':
            values['do_not_use_stop_stage'] = 'false'
        else:
            values['do_not_use_stop_stage'] = 'true'

        fp = open(fn, "w")
        msj_template = \
            """
            task {
              set_family = {
                 desmond = {
                    checkpt = {
                       write_last_step = false
                    }
                 }
                 simulate = {
                    trajectory = {
                       center = "solute"
                    }
                 }
              }
              task = "desmond:auto"
            }


            concatenate {
              simulate = [
                 {ensemble = {
                     brownie = {
                        delta_max = 0.1
                     }
                     class = NVT
                     method = Brownie
                  }
                  restrain = {
                     atom = solute_heavy_atom
                     force_constant = 50.0
                  }
                  temperature = 10.0
                  time = %(equib1_time)f
                  timestep = [0.001 0.001 0.003 ]
                  title = "Brownian Dynamics NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 100ps"
                 }

                 {effect_if = [
                     ["==" -gpu "@*.*.jlaunch_opt[-1]" ]
                     "ensemble.method = Langevin"
                  ]
                  eneseq = {
                     interval = 0.3
                  }
                  ensemble = {
                     class = NVT
                     method = Berendsen
                     thermostat = {
                        tau = 0.1
                     }
                  }
                  randomize_velocity = {
                     interval = 1.0
                     seed     = %(seed1)d
                  }
                  restrain = {
                     atom = solute_heavy_atom
                     force_constant = 50.0
                  }
                  temperature = 10.0
                  time = %(equib2_time)f
                  timestep = [0.001 0.001 0.003 ]
                  title = "NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 12ps"
                  trajectory = {
                     interval = 24.0
                  }
                 }

                 {effect_if = [
                     ["==" -gpu "@*.*.jlaunch_opt[-1]" ]
                     "ensemble.method = Langevin"
                  ]
                  eneseq = {
                     interval = 0.3
                  }
                  ensemble = {
                     barostat = {
                        tau = 50.0
                     }
                     class = NPT
                     method = Berendsen
                     thermostat = {
                        tau = 0.1
                     }
                  }
                  randomize_velocity = {
                     interval = 1.0
                     seed     = %(seed1)d
                  }
                  restrain = retain
                  temperature = 10.0
                  time = %(equib3_time)f
                  title = "NPT, T = 10 K, and restraints on solute heavy atoms, 12ps"
                 }
              ]
            }

            simulate {
              %(solvate_pocket)s
              ensemble = NVT
              gcmc = {
                 gcmc_region = {
                    region_buffer = 6.000000
                 }
                 interval = 1.0
                 moves = {
                    moves_per_cycle = 5000
                 }
                 solvent = {
                 }
              }
              restrain = {
                 atom = solute_heavy_atom
                 force_constant = 50.0
              }
              time = %(equib4_time)f
              title = "GCMC Solvate Pocket muVT, T=300K, restraints on heavy atoms"
            }

            simulate {
              %(solvate_pocket)s
              ensemble = NVT
              gcmc = {
                 gcmc_region = {
                    region_buffer = 6.000000
                 }
                 interval = 1.0
                 moves = {
                    moves_per_cycle = 5000
                 }
                 solvent = {
                 }
              }
              restrain = none
              time = %(equib5_time)f
              title = "GCMC Solvate Pocket muVT, T=300K, no restraints"
            }

            simulate {
              ensemble = {
                 brownie = {
                    delta_max = 0.1
                 }
                 class = NVT
                 method = Brownie
              }
              restrain = {
                 atom = solute_heavy_atom
                 force_constant = 50.0
              }
              temperature = 300.0
              time =%(equib6_time)f
              timestep = [0.001 0.001 0.003 ]
              title = "Brownian Dynamics NVT, T = 300 K, small timesteps, and restraints on solute heavy atoms"
            }

            ##########
            # Unbiased MD production phase
            #########
            simulate {
              effect_if = [
                 ["==" "-gpu" "@*.*.jlaunch_opt[-1]" ]
                 "ensemble.method = Langevin"
              ]
              eneseq = {
                 interval = 0.3
              }
              %(restrain_info)s
              ensemble = {
                 #barostat = {
                 #   tau = 2.0
                 #}
                 class = "NVT"
                 method = "Berendsen"
                 thermostat = {
                    tau = 0.1
                 }
              }
              meta = {
                 #bias potential height in kcal/mol--about 1/10th k_B T
                 height=0.0
                 #biasing interval in ps
                 interval=1
                 #output file of cv values
                 cv_name = $JOBNAME.cvseq
                 #output file of bias potential added at each bias time
                 name=metadynamics_outfile.dat

                 #define cvs
                 cv = [ { type = rmsd
                          #rmsd of all heavy atoms in the ligand (ligand target assumed to be chain Z)
                          atom = ["(%(cv_asl)s) AND NOT (atom.ele H)"]
                          #this is the default--see what happens
                          width = 0.0
                    }]
                 # well tempering parameter--modified height
                 kTemp=.6
              }
              time = %(equib7_time)f
              trajectory = {
                 interval = 200.0
              }
            }

            stop {
                should_skip = %(do_not_use_stop_stage)s
            }

            ##########
            # Metadynamics production phase
            #########
            simulate {
              effect_if = [
                 ["==" "-gpu" "@*.*.jlaunch_opt[-1]" ]
                 "ensemble.method = Langevin"
              ]
              eneseq = {
                 interval = 0.3
              }
              ensemble = {
                 #barostat = {
                 #   tau = 2.0
                 #}
                 class = "NVT"
                 method = "Berendsen"
                 thermostat = {
                    tau = 0.1
                 }
              }
              meta = {
                 #bias potential height in kcal/mol--about 1/10th k_B T
                 height=%(height)f
                 #biasing interval in ps
                 interval=1
                 #output file of cv values
                 cv_name = $JOBNAME.cvseq
                 #output file of bias potential added at each bias time
                 name=metadynamics_outfile.dat

                 #define cvs
                 cv = [ { type = rmsd
                          #rmsd of all heavy atoms in the ligand (ligand target assumed to be chain Z)
                          atom = ["(%(cv_asl)s ) AND NOT (atom.ele H)"]
                          #this is the default--see what happens
                          width = 0.02
                    }]
                 # well tempering parameter--modified height
                 kTemp=%(ktemp)f
              }
              %(hmr_timestep)s
              time = %(prod_time)f
              trajectory = {
                 interval = 200.0
              }
            }
            """
        msj_template = textwrap.dedent(msj_template)
        msj_template = msj_template % (values)
        fp.write(msj_template)
        fp.close()

    @staticmethod
    def cleanup_tarfile(tar_obj, extract_dir):
        """Deletes the extracted files from a tarfile.
           Assumes the tarfile only contains files and
           directories. Nothing exotic like a block device
           or a character device.
        """
        # First remove all the files, skipping directories
        for tar_info in tar_obj.getmembers():
            if tar_info.isfile():
                file = os.path.join(extract_dir, tar_info.name)
                os.remove(file)

        # Now remove all the directories
        # starting from the most nested dir
        dirs = [
            tar_file.name
            for tar_file in tar_obj.getmembers()
            if tar_file.isdir()
        ]
        dirs.sort(key=lambda dir: dir.count(os.path.sep), reverse=True)
        for dir in dirs:
            try:
                os.rmdir(os.path.join(extract_dir, dir))
            except OSError:
                # If the workdir resides on an NFS, the python process
                # may create a .nfs* file which would prevent
                # rmdir from completing. This excepts that error
                # leaving the otherwise empty directory intact.
                pass

    def get_md_equil_ct(self, ligand_asl, cmsout_fn, cmstraj_fn, ipose, itrial,
                        writer):
        """Extracts the MD equil ct and preps it for alignment.
           The prepped ct is written to the {writer}."""
        _, cms_model = topo.read_cms(cmsout_fn)
        trj = traj.read_traj(cmstraj_fn)
        # First frame from MtD production stage
        # is the output structure from
        # MD equilibration
        frame = trj[0]
        frame_ct = topo.update_ct(cms_model.fsys_ct, cms_model, frame)
        frame_ct.title = "MD Equil Pose {} Trial {}".format(
            ipose + 1, itrial + 1)
        frame_ct = self.set_ligand_to_mol_1(ligand_asl, frame_ct)
        frame_ct.property[self.POSE_TAG] = ipose
        frame_ct.property[self.TRIAL_TAG] = itrial
        frame_ct.property['s_m_subgroupid'] = 'Pose{}'.format(ipose + 1)
        frame_ct.property['b_m_subgroup_collapsed'] = True
        del frame_ct.property['s_chorus_trajectory_file']
        writer.append(frame_ct)

    def get_md_equil_rmsd(self, opts, cts, align_dir, md_equil_prealigned_mae):
        """Runs aligning_binding_sites on the cts in md_equil_prealigned_mae.
           Assumes that each ct has except for the first ct, which is the
           reference ct, has the properties self.POSE_TAG and self.TRIAL_TAG set.
           Using these properties, a list of lists will be returned.
           The format will be:
                 Pose 1 Trials       Pose 2 Trials       etc.
           [ [Trial1, Trial2, ...], [Trial1, Trial2, ..] ]

           This method will also remove the align_dir but extract
           out the aligned output mae and register it as
           a jobcontrol output file.
        """
        # md_equil_rmsds is a matrix of the form: md_equil_rmsds[ipose][itrial]
        # Where ipose and itrial are 0-based.
        # Initializing the rmsd to a value of -1, which if remained
        # unchanged, indicates that an RMSD value was not obtained.
        # This may occur if align_binding_sites failed.
        md_equil_rmsds = [
            [-1 for j in range(opts.ntrial)] for i in range(len(cts))
        ]
        cwd = os.getcwd()
        os.chdir(align_dir)
        aligned_output_mae = opts.jobname + '_md_equil_aligned.maegz'
        cmd = [
            'align_binding_sites',
            os.path.basename(md_equil_prealigned_mae), '-l', '1', '-JOBNAME',
            'md_equil', '-o', aligned_output_mae, '-NOJOBID'
        ]
        try:
            std_output = check_output(
                cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Align binding sites error:")
            print(e.output)
            raise

        # The +1 is for a reference ct
        n_expected_cts = len(cts) * opts.ntrial + 1
        n_output_cts = structure.count_structures(aligned_output_mae)
        if n_expected_cts != n_output_cts:
            print("Warning: Align binding sites failed to align all trials")
            print("Unexpected number of output structures")
            print("Expected: {}".format(n_expected_cts))
            print("Observed: {}\n".format(n_output_cts))
            if psp_util.was_run_with_debug():
                print("\nalign_binding_sites stdout:")
                print(std_output)
                print('\n')

        rmsd_asl = self.INTERNAL_LIG_ASL
        if opts.ligand_tail_for_md_equil_rmsd:
            smarts_string = opts.ligand_tail_for_md_equil_rmsd
            rmsd_asl += ' AND NOT ("SMARTS. {}")'.format(smarts_string)
        for aligned_ct in structure.StructureReader(
                aligned_output_mae, index=2):
            md_equil_ligand_iatoms = evaluate_asl(aligned_ct, rmsd_asl)
            ipose = aligned_ct.property[self.POSE_TAG]
            itrial = aligned_ct.property[self.TRIAL_TAG]
            input_ct_ligand_iatoms = evaluate_asl(cts[ipose], rmsd_asl)
            ligand_rmsd = calculate_in_place_rmsd(
                aligned_ct,
                md_equil_ligand_iatoms,
                cts[ipose],
                input_ct_ligand_iatoms,
                use_symmetry=True)
            md_equil_rmsds[ipose][itrial] = ligand_rmsd

        shutil.copyfile(aligned_output_mae, os.path.join(
            cwd, aligned_output_mae))
        os.chdir(cwd)
        self.addOutputFile(os.path.join(cwd, aligned_output_mae))

        shutil.rmtree(align_dir)

        return md_equil_rmsds

    @staticmethod
    def set_ligand_to_mol_1(ligand_asl, ct):
        """Returns a new ct with the ligand is molecule 1"""
        ATOM_PROP = "i_psp_temp_orig_atom_number"
        for atom in ct.atom:
            atom.property[ATOM_PROP] = int(atom)
        lig_iatoms = analyze.evaluate_asl(ct, ligand_asl)
        if not lig_iatoms:
            error = "Ligand ASL ('{}') matches no atoms".format(ligand_asl)
            raise RuntimeError(error)
        other_ct = ct.copy()
        other_ct.deleteAtoms(lig_iatoms)
        lig_ct = ct.extract(lig_iatoms, copy_props=True)
        ct = lig_ct.merge(other_ct, copy_props=True)
        return ct


if __name__ == "__main__":
    BindingPoseMetadynamics(sys.argv[0]).launch(sys.argv[1:])
