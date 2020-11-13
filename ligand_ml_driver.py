"""
Runs ligand_ml under JobControl.

This driver script is a simple wrapper for running ligand_ml under JobControl,
and performs three tasks:
    - build and train a ligand_ml (DeepChem/AutoQSAR) model
    - predict on new data using an existing pre-trained model
    - generate a report of an existing pre-trained model

The build task will generate a <model>.qzip file, which is a required input for
the predict and report tasks. Accepted input file formats are MAE, SDF, and CSV.
However, note that only MAE and CSV file formats are accepted for prediction
outputs as of now.

Additional descriptors / features may be specified in a text file using the -x
flag, where each descriptor is specified on a new line.

The build task may be run in serial or parallel mode depending on the
-NUM_SUBJOBS argument to increase the number of ML models trained and evaluated
from the random hyper-parameter search space.

Usage Examples:

BUILD:

    MAE input:
    $SCHRODINGER/run ligand_ml_driver.py model.qzip -build -regression -y
    r_m_log-solubility -i example.mae.gz -smiles_col smiles -JOB BuildTask_1
    -HOST localhost:1 -NUM_SUBJOBS 1

    The target / y value must be a Maestro property, and must exist for each
    structure in the given input file.

    SDF input:
    $SCHRODINGER/run ligand_ml_driver.py model.qzip -build -regression -y
    r_sd_log-solubility -i example.sdf -smiles_col smiles

    Note that all properties referred in the SDF input must be in Maestro
    structure property format, as given here by 'r_sd_log-solubility'.

    CSV input:
    $SCHRODINGER/run ligand_ml_driver.py model.qzip -build -regression -y
    log-solubility -i example.csv -smiles_col smiles

    For CSV files, a header line must be included and the SMILES column must be
    denoted as 'SMILES' or must be set using the '-smiles_col' argument as shown
    above.

PREDICT:

    $SCHRODINGER/run ligand_ml_driver.py model.qzip -predict -i example.mae
    -pred predict_output.mae

REPORT:

    $SCHRODINGER/run ligand_ml_driver.py model.qzip -report

Copyright Schrodinger LLC, All Rights Reserved.
"""

import argparse
import copy
import csv
import json
import multiprocessing
import numpy as np
import os
import subprocess
import sys
import time
import zipfile

from schrodinger import structure
from schrodinger.job import driver_decorator
from schrodinger.job import jobcontrol
from schrodinger.job import launchapi
from schrodinger.job import queue
from schrodinger.utils import env as env_utils
from schrodinger.utils import fileutils
from schrodinger.utils import log
from schrodinger.utils import license
from schrodinger.structutils import smiles

try:
    import zlib

    compression = zipfile.ZIP_DEFLATED
except ImportError:
    compression = zipfile.ZIP_STORED

# Logging
logger = log.get_output_logger(__file__)

# For job control
SCHRODINGER = os.getenv("SCHRODINGER")
SCHRODINGER_RUN = os.path.join(SCHRODINGER, "run")

# Driver name
LIGAND_ML_DRIVER = "ligand_ml_driver.py"
# Non ligand_ml arguments that ligand_ml_driver allows
NON_ML_ARGS = {'JOB', 'report', 'o', 'details'}
# Number of ligands per prediction job
STRUCTURES_PER_JOB = 64000


class LigandMLDriver:

    def __init__(self, args):
        """
        Initializes the driver from command line arguments.

        :param args: argument namespace with command line options
        :type args: argparse.Namespace
        """

        self.args = args
        self.run_folder = None
        self.original_pred_file = None
        self.original_pred_input_file = None
        self.args.descriptors = self.get_additional_descriptors()

    def run(self):
        """
        Runs the appropriate task of either building, predicting, or
        generating a report based on the provided arguments.
        """

        # Confirm appropriate platform
        if not sys.platform.startswith('linux') and not self.args.report:
            sys.exit('ligand_ml jobs must be submitted to a Linux host.')

        if self.args.build:
            self.build()
        elif self.args.predict:
            self.predict()
        elif self.args.report:
            self.report()
        else:
            raise ValueError('Invalid input args given.')

        # FIXME: Do we really need this file?
        cmd = 'echo $(date) >> ' + get_jobname(self.args) + '.okay'
        subprocess.call(cmd, shell=True)

    def build(self):
        """
        Given the input data and requested property, build a new ligandML model.

        Note: This driver converts all inputs to SDFs so that they may be
        parsed by ligandML. This is a known limitation that will be resolved in
        ML-512.
        """

        args = self.args
        self.run_folder = self._make_run_folder()

        # input structure file
        args.i = self.convert_file(args.i, task='input_build')

        # input holdout file
        if args.holdout:
            args.holdout = self.convert_file(args.holdout, task='input_holdout')

        cmd = self.create_interface_cmd()
        self.run_interface(cmd)
        self.cleanup_interface_run()

    def predict(self):
        """
        Using an existing ligandML model, predict on an unseen set of samples.

        Note: This driver converts all inputs to SDFs so that they may be
        parsed by ligandML. This is a known limitation that will be resolved in
        ML-512.
        """

        args = self.args
        self.run_folder = self._make_run_folder()

        # input structure file
        self.original_pred_input_file = args.i
        args.i = self.convert_file(args.i, task='input_predict')

        # output prediction file
        if not fileutils.is_csv_file(args.pred):
            self.original_pred_file = args.pred
            out_file = os.path.join(self.run_folder, 'temp_output.csv')
            args.pred = out_file

        cmd = self.create_interface_cmd()
        self.run_interface(cmd)
        self.merge_prediction_output()
        self.cleanup_interface_run()

    def report(self):
        """
        Generate a report summary of the requested model. Note that this task
        does not call ligand_ml_interface.py.
        """

        report_json = report(self.args)
        if self.args.o:
            with open(self.args.o, 'w') as fh:
                json.dump(report_json, fh, sort_keys=True, indent=2)
        else:
            print(json.dumps(report_json, sort_keys=True, indent=2))

    def merge(self, models):
        """
        Merge together several ligand_ml runs into a single result.

        :param models: .qzip file paths to merge together.
        :type models: list[str]
        """

        self.args.merge = models
        self.build()

    def _make_run_folder(self):
        """
        Make a folder for running ligand_ml in the same directory as where the
        model lives, with the same name

        :return: run folder file path
        :rtype: str
        """

        model = self.args.model
        run_folder = os.path.splitext(model)[0]
        fileutils.force_rmtree(run_folder)  # remove if it already exists
        os.mkdir(run_folder)
        return run_folder

    def create_interface_cmd(self):
        """
        Create command line arguments for calling ligand_ml_interface.py.

        Note: In order to limit tensorflow to a single processor, `taskset`
        must be called prior to executing `ligand_ml_interface.py`.

        :return: command line argument list for ligand_ml_interface.py
        :rtype: list[str]
        """

        # Get args as a dict and exclude args invalid for ligand_ml_interface.py
        common_args_dict = get_dict_from_namespace(
            self.args, exclude_args=NON_ML_ARGS)

        # Convert args to JSON str for use in ligand_ml_interface.py
        cmd_args = json.dumps(common_args_dict)
        cmd = [SCHRODINGER_RUN, "ligand_ml_interface.py", cmd_args]

        # Setup taskset command so we can ensure tensorflow doesn't get greedy
        procs = str(np.random.randint(multiprocessing.cpu_count()))
        logger.info("Core for ligand_ml_interface.py set to " + procs)
        cmd = ["taskset", "-c", procs] + cmd

        return cmd

    def run_interface(self, cmd):
        """
        Run ligand_ml_interface command.

        :param cmd: command line arguments to invoke ligand_ml_interface with.
        :type cmd: list[str]
        """

        env = os.environ.copy()

        if 'SCHRODINGER_CUDA_VISIBLE_DEVICES' in os.environ:
            gpus = os.environ['SCHRODINGER_CUDA_VISIBLE_DEVICES']
            env['CUDA_VISIBLE_DEVICES'] = gpus
            print("SCHRODINGER_CUDA_VISIBLE_DEVICES set to " + gpus)
        elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
            env['CUDA_VISIBLE_DEVICES'] = "INVALID"

        env_utils.add_cuda_stubs_for_tensorflow(env)
        ret = subprocess.call(cmd, env=env)
        if ret != 0:
            sys.exit("ligand_ml_interface.py failed")

    def cleanup_interface_run(self):
        """
        General catch-all cleanup method after a ligand_ml_interface run.
        """

        fileutils.force_rmtree(self.run_folder)

    def convert_file(self, in_file, task=''):
        """
        Convert given file so it can be appropriately parsed by ligand_ml:
        MAE file -> SD file + target and additional features format
        SD file -> SD file + target and additional features format
        CSV file -> CSV file + column name dictionary

        :param in_file: the input Maestro / SD / CSV file to format.
        :type in_file: str

        :param task: additional string to denote task type to add to output
            file name.
        :type task: str

        :return: the formatted output file path.
        :rtype: str
        """
        if fileutils.is_maestro_file(in_file) or fileutils.is_sd_file(in_file):
            out_file = os.path.join(self.run_folder, f'temp_{task}.sdf')
            self.convert_to_sd_file_type(in_file, out_file)
            return out_file
        elif fileutils.is_csv_file(in_file):
            # no conversion necessary for CSV files, instead construct columns
            self.args.columns = {
                'smiles': self.args.smiles_col,
                'response': self.args.y,
                'extra': self.args.descriptors
            }
            return in_file
        else:
            raise ValueError(f'Invalid File Format: {in_file}. '
                             'Must be of type MAE, CSV, or SDF.')

    def convert_to_sd_file_type(self, in_file, out_file):
        """
        Formats the structure properties to match what ligand_ml expects as
        input for the y values and additional features.

        :param in_file: the input Maestro / SD file to format.
        :type in_file: str

        :param out_file: the output formatted SDF file.
        :type out_file: str
        """
        # Temporary measure until MAE support is added in LigandML (ML-512)
        if not in_file:
            return
        sts = []
        prop_map = {}
        descriptor_map = {
            descriptor: 's_sd_additional\\_feature\\_%d' % idx
            for idx, descriptor in enumerate(self.args.descriptors)
        }
        prop_map.update(descriptor_map)
        if self.args.y:
            prop_map[self.args.y] = 's_sd_target\\_value\\_0'
        for st in structure.StructureReader(in_file):
            for prop, new_prop in prop_map.items():
                st.property[new_prop] = str(st.property[prop])
            sts.append(st)
        with structure.StructureWriter(out_file) as writer:
            writer.extend(sts)

    def get_additional_descriptors(self):
        """
        Read in additional descriptors if specified.

        :return: List of descriptors.
        :rtype: List[str]
        """
        if not self.args.x:
            return []
        with open(self.args.x, 'r') as fh:
            desc = fh.read()
        return desc.strip().split('\n')

    def merge_prediction_output(self):
        """
        Merges Ligand ML's predicted output in CSV format with the input
        structures to produce the Maestro format output, where the prediction
        scores and uncertainties are added as structure properties.
        """
        in_file = self.original_pred_input_file
        out_file = self.original_pred_file
        if not out_file:
            return  # early return if .csv output requested
        # if .csv not requested, .mae format generated for output
        score_prop = 'r_m_' + self.args.prop + '_score'
        uncertainty_prop = 'r_m_' + self.args.prop + '_uncertainty'
        with structure.StructureWriter(out_file) as writer:
            with open(self.args.pred, 'r') as csv_fh:
                reader = csv.reader(csv_fh)
                st_reader = structure.StructureReader(in_file)
                for row, st in zip(reader, st_reader):
                    st.property[score_prop] = float(row[1])
                    st.property[uncertainty_prop] = float(row[2])
                    writer.append(st)


# =============================================================================
# Serial & Parallel Run Functions
# =============================================================================


def check_prediction_split(args):
    """
    Checks for number of ligands in prediction input file:
    if the number of ligands are < STRUCTURES_PER_JOB we resort to serial execution on a single host
    if the number of ligands > STRUCTURES_PER_JOB we use multiple hosts if available to fill up each
    with (up to) the value set to STRUCTURES_PER_JOB

    :param: args: command line arguments
    :type args: argparse.Namespace
    """
    num_ligands = get_num_ligands(args.i)
    if num_ligands < STRUCTURES_PER_JOB:
        run_serial(args)
    else:
        run_distributed_predict(args)
    logger.info('LigandML driver prediction completed')


def get_num_ligands(in_file):
    """
    Returns the number of ligands depending on the file extension

    :param: in_file: path to check the number of rows or structures
    :type in_file: str

    :return: number of structures
    :rtype: int
    """
    if fileutils.is_maestro_file(in_file) or fileutils.is_sd_file(in_file):
        return len([x for x in structure.StructureReader(in_file)])
    elif fileutils.is_csv_file(in_file):
        with open(in_file, 'r') as fh:
            num_ligands = len(fh.readlines()) - 1  # omitting header
        return num_ligands


def sd_mae_split(args, structures_per_job):
    """
    Splits .mae input file across multiple files,
    depending on the number of ligands and available
    resources

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: structures_per_job: permissible number of structures
    per subjob, depending on split
    :type: int

    :return: subjob input prediction files
    :rtype: list
    """
    st_reader = structure.StructureReader(args.i)
    batch = []
    batches = []
    # generate batches of structures
    for index, st in enumerate(st_reader):
        if index % structures_per_job == 0 and batch:
            batches.append(batch)
            batch = []
        batch.append(st)
    if batch:
        batches.append(batch)
    # write each batch separately to a new input file
    prefix, ext = os.path.splitext(args.i)
    new_predict_inputs = []
    for index, batch in enumerate(batches):
        predict_input_sub = "{}_input_sub_{}{}".format(prefix, index + 1, ext)
        new_predict_inputs.append(predict_input_sub)
        with structure.StructureWriter(predict_input_sub) as writer:
            writer.extend(batch)
    return new_predict_inputs


def csv_split(args, structures_per_job):
    """
    Splits .csv input file across multiple files,
    depending on the number of ligands and available
    resources

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: structures_per_job: permissible number of structures
    per subjob, depending on split
    :type: int

    :return: subjob input prediction files
    :rtype: list
    """
    subjob_file = None
    new_input_files = []
    prefix, ext = os.path.splitext(args.i)
    batch_number = 1  # set to 1 for user readability
    with open(args.i, "r") as csv_fh:
        header = csv.DictReader(csv_fh).fieldnames
        for index, row in enumerate(csv_fh):
            if index % structures_per_job == 0:
                if subjob_file:
                    subjob_file.close()
                subjob_file = "{}_sub_{}{}".format(prefix, batch_number, ext)
                new_input_files.append(subjob_file)
                subjob_file = open(subjob_file, "w")
                writer = csv.DictWriter(subjob_file, fieldnames=header)
                writer.writeheader()
                batch_number += 1
            subjob_file.write(row)
        if subjob_file:
            subjob_file.close()
    return new_input_files


def _check_optimization_logic(n_jobs, rows):
    """
    check to enable scenario 1 distributed execution
    where each subjob can handle the optimal number of
    structures
    :param: n_jobs: number of jobs specified by args.NUM_SUBJOBS
    :type n_jobs: int

    :param: rows: number of ligands
    :type rows: int
    :return: whether to optimize
    :rtype: bool
    """
    optimal_subjobs = rows // STRUCTURES_PER_JOB
    # check for available resource to handle residual structures
    if rows % STRUCTURES_PER_JOB and n_jobs >= optimal_subjobs + 1:
        return True
    # if no residual structures, check for enough resources
    elif not rows % STRUCTURES_PER_JOB and n_jobs >= optimal_subjobs:
        return True
    return False


def split_structures(args):
    """
    Scenario 1: Considering we have enough resources for optimal execution
    Scenario 2: Ligands are distributed evenly across hosts

    :param: args: command line arguments
    :type args: argparse.Namespace

    :return: splitter method that returns subjob input files
    :rtype: list
    """
    rows = get_num_ligands(args.i)
    n_jobs = args.NUM_SUBJOBS
    # default structures_per_job, each resource is given an even load
    structures_per_job = rows // n_jobs
    # check if we can optimize execution (scenario 1)
    optimize = _check_optimization_logic(n_jobs, rows)
    if optimize:
        structures_per_job = STRUCTURES_PER_JOB
    if fileutils.is_maestro_file(args.i) or fileutils.is_sd_file(args.i):
        return sd_mae_split(args, structures_per_job)
    elif fileutils.is_csv_file(args.i):
        return csv_split(args, structures_per_job)
    else:
        raise ValueError("File type {} cannot be supported".format(
            args.i.splitext()[1]))


def create_subjob_output_files(filename, num_subjobs):
    """
    Generic method to create n-subjob files where n is num_subjobs,
    using the prefix and extension of the original filename

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: filename: original file
    :type: filename: str

    :param: num_subjobs: allowed subjobs depending on host
    provisioning and number of structures
    :type: num_subjobs: int

    return: subjob output files
    :rtype: list
    """
    prefix, ext = os.path.splitext(filename)
    output_files = []
    for i in range(num_subjobs):
        output_subjob = "%s_sub_%d" % (prefix, i + 1)
        output_files.append(output_subjob)

    return [p + ext for p in output_files]


def combine_mae_format(args, subjob_predict_outputs):
    """
    if args.pred == .mae this function generates
    an .mae file that merges all prediction outputs

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: subjob_predict_outputs: file paths to each subjob output
    :type: subjob_predict_outputs: list of strings
    """
    with structure.StructureWriter(args.pred) as writer:
        for out_file in subjob_predict_outputs:
            structs = list(structure.StructureReader(out_file))
            writer.extend(structs)


def combine_csv_format(args, subjob_predict_outputs):
    """
    if args.pred == .csv this function generates
    a .csv file that merges all prediction outputs

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: subjob_predict_outputs: file paths to each subjob output
    :type: subjob_predict_outputs: list of strings
    """
    header = ['SMILES', 'score', 'std']
    with open(args.pred, 'w') as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()
        for subjob_output in subjob_predict_outputs:
            with open(subjob_output, "r") as fh:
                for row in fh:
                    fout.write(row)


def combine_prediction_outputs(args, subjob_predict_outputs):
    """
    If the user-defined arg.pred ends in .csv, we merge
    the output .csv files generated by all the ligand_ml
    prediction subjobs into a single .csv file.

    If the user-defined arg.pred ends in .mae format, we
    merge the already converted .mae files from the prediction
    subjobs into a single .mae output file

    :param: args: command line arguments
    :type args: argparse.Namespace

    :param: subjob_predict_outputs: file paths to each subjob output
    :type: subjob_predict_outputs: list
    """
    if fileutils.is_maestro_file(args.pred) or fileutils.is_sd_file(args.pred):
        combine_mae_format(args, subjob_predict_outputs)
    elif fileutils.is_csv_file(args.pred):
        combine_csv_format(args, subjob_predict_outputs)


def run_serial(args):
    """
    Runs a single ligand_ml job based on the provided arguments.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace
    """

    # Tampering with licensing is a violation of the license agreement
    token = license.License(license.licensing.AUTOQSAR_MAIN, 1, force=True)
    if not token.isValid():
        sys.exit("AUTOQSAR license token is unavailable. Contact "
                 "help@schrodinger.com to request additional licenses.")
    LigandMLDriver(args).run()


def run_distributed_predict(args):
    """
    Sets up a distributed job by launching subjobs using JobDJ and combines
    results.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace
    """
    logger.info('\n Starting Prediction LigandML Distributed...')
    jobname = get_jobname(args)
    subjob_predict_inputs = split_structures(args)
    subjob_names = get_subjob_names(len(subjob_predict_inputs), jobname)
    subjob_predict_outputs = create_subjob_output_files(
        args.pred, len(subjob_predict_inputs))
    common_argv = get_subjob_pred_argv(args)
    jobdj = queue.JobDJ(
        default_max_retries=1, verbosity="normal", max_failures=queue.NOLIMIT)

    for name, input_pred, output_pred in zip(
            subjob_names, subjob_predict_inputs, subjob_predict_outputs):
        cmd = [
            LIGAND_ML_DRIVER, args.model, '-JOB', name, '-SUBJOB', '-i',
            input_pred, '-pred', output_pred
        ] + common_argv
        jobdj.addJob(cmd)
    jobdj.run()

    combine_prediction_outputs(args, subjob_predict_outputs)
    combine_log_files(subjob_names, logger)
    combine_out_files(subjob_names, '.okay', jobname)
    combine_out_files(subjob_names, '.errors', jobname)


def run_distributed(args, num_subjobs):
    """
    Sets up a distributed job by launching subjobs using JobDJ and combines
    results.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :param num_subjobs: number of subjobs to run in parallel.
    :type num_subjobs: int
    """
    jobname = get_jobname(args)
    subjob_names = get_subjob_names(num_subjobs, jobname)
    model_names = get_model_subjob_names(num_subjobs, args.model)
    common_argv = get_subjob_argv(args)
    random_seed = str(np.random.randint(int(time.time())))
    args.seed = random_seed

    jobdj = queue.JobDJ(
        default_max_retries=1, verbosity="normal", max_failures=queue.NOLIMIT)

    for name, model in zip(subjob_names, model_names):
        cmd = [
            LIGAND_ML_DRIVER, model, '-JOB', name, '-SUBJOB', '-seed',
            random_seed
        ] + common_argv
        jobdj.addJob(cmd)

    jobdj.run()

    combine_log_files(subjob_names, logger)
    combine_out_files(subjob_names, '.okay', jobname)
    combine_out_files(subjob_names, '.errors', jobname)
    logger.info('\nMerging ligand_ml models:')
    combine_subjobs(args, model_names)


def combine_subjobs(args, subjob_models):
    """
    Combine all the distributed build models into a single model for output.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :param subjob_models: Names of the model files to be combined.
    :type subjob_models: list[str]
    """

    driver = LigandMLDriver(args)
    driver.merge(subjob_models)


def combine_log_files(subjobs, logger):
    """
    Concatenates the contents of subjob log files.

    :param subjobs: subjob names
    :type subjobs: list(str)

    :param logger: logger to which concatenated log files are to be written
    :type logger: Logger
    """

    logger.info("Combining subjob log files")
    for subjob in subjobs:
        logfile = subjob + ".log"
        if os.path.isfile(logfile):
            logger.info("\n*** Contents of %s ***\n" % logfile)
            with open(logfile, 'r') as fh:
                for line in fh:
                    logger.info(line.rstrip())
            logger.info("\n*** End of %s ***" % logfile)
        else:
            logger.info("\n*** %s is not a file ***\n" % logfile)


def combine_out_files(subjobs, extension, model_name):
    """
    Concatenates the contents of subjob output files.

    :param subjobs: subjob names
    :type subjobs: list(str)

    :param extension: output file extensions.
    :type extension: str

    :param model_name: the model name to construct the output file name.
    :type model_name: str
    """

    outfile = model_name + extension
    with open(outfile, 'w') as out_fh:
        out_fh.write("Combining subjob %s files" % extension)
        for subjob in subjobs:
            sub_outfile = subjob + extension
            if os.path.isfile(sub_outfile):
                out_fh.write("\n*** Contents of %s ***\n" % sub_outfile)
                with open(sub_outfile, 'r') as fh:
                    for line in fh:
                        out_fh.write(line.rstrip())
                out_fh.write("\n*** End of %s ***" % sub_outfile)
            else:
                out_fh.write("\n*** %s is not a file ***\n" % sub_outfile)


# =============================================================================
# Command line Parser
# =============================================================================


def get_parser():
    """
    Creates argument parser for deep learning ligand_ml job.

    :return: configured argument parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required args
    parser.add_argument(
        "model",
        metavar="<model>.qzip",
        help="QSAR model archive. Will be created if building models; must "
        "already exist if predicting or generating a report.")

    # Add argument groups
    add_build_arguments(parser)
    add_predict_arguments(parser)
    add_common_arguments(parser)
    add_file_arguments(parser)
    add_report_arguments(parser)
    add_jobcontrol_arguments(parser)
    return parser


def add_build_arguments(parser):
    """
    Options to build new ligand_ml models.

    :param parser: to add build args to.
    :type parser: argparse.ArgumentParser
    """

    group = parser.add_argument_group('build new model arguments',
                                      'Options to build new ligand_ml models.')
    group.add_argument(
        "-build",
        action="store_true",
        help="Build new model from the provided structures and activity "
        "values.")
    group.add_argument(
        "-y",
        metavar="<propname>",
        help="Activity property name. May be an integer, real or string.")
    group.add_argument(
        "-classification",
        action="store_true",
        help="Build classification model from an integer, real or string "
        "property.")
    group.add_argument(
        "-regression",
        action="store_true",
        help="Build continuously-valued numeric model from an integer or real "
        "activity property.")
    group.add_argument(
        "-split",
        metavar="<fract>",
        help="Fraction of compounds to assign to training set vs holdout set "
        "when building models. Must be less than 1.0 (default: 0.75).",
        type=float,
        default=0.75)
    group.add_argument(
        '-split_type',
        metavar='<random OR scaffold>',
        help='Type of split to apply when generating the training and holdout '
        'set. Must be either "random" or "scaffold" (default: "random"',
        default='random')
    group.add_argument(
        "-holdout",
        metavar="<holdoutFile>",
        help="After building models, validate them by making predictions on "
        "the structures in <holdoutFile>. This file must contain an observed "
        "activity value for every structure. Must be of type MAE, CSV, or SDF.")
    group.add_argument(
        "-time",
        metavar="<hours>",
        help="Floating point time limit for training deep learning models.",
        type=float,
        default=4.0)
    group.add_argument(
        "-seed",
        help="Random seed to set to ensure identical training and holdout "
        "data sets. This value will only be used if a holdout file has "
        "not been given by the flag -holdout",
        type=int,
        default=None)


def add_predict_arguments(parser):
    """
    Options to predict using existing ligand_ml models.

    :param parser: to add model predict args to.
    :type parser: argparse.ArgumentParser
    """

    group = parser.add_argument_group(
        'predict using existing models arguments',
        'Options to predict using existing ligand_ml models.')
    group.add_argument(
        "-predict",
        action="store_true",
        help="Evaluate one or more models in the provided QSAR archive on the "
        "input structures.")
    group.add_argument(
        "-pred",
        metavar="<predfile>",
        help="Output file for predictions. Must be a Maestro file or CSV for "
        "outputting predictions.",
        default="predict_output.mae")


def add_common_arguments(parser):
    """
    Common input data manipulation arguments.

    :param parser: to add common args to.
    :type parser: argparse.ArgumentParser
    """

    group = parser.add_argument_group(
        'common arguments', 'Common input data manipulation arguments.')
    group.add_argument(
        "-log", action="store_true", help="Add log transformer to models.")
    group.add_argument(
        "-prop",
        metavar="<string>",
        help="Incorporate <string> into prediction property names.",
        default="ligand_ml")
    group.add_argument(
        '-smiles_col',
        metavar="<string>",
        help="Denotes the SMILES column name for CSV file inputs.",
        default="SMILES")


def add_file_arguments(parser):
    parser.add_argument(
        "-i",
        metavar="<infile>",
        help="Input file with structures and activities. Must be of type MAE, "
        "CSV, or SDF. Note, if a CSV file is given, it must include a "
        "header line, and the SMILES column must be denoted as 'SMILES' "
        "or must be set using the `-smiles_col` optional arg.")
    parser.add_argument(
        "-o",
        metavar="<outfile>",
        help="Write output to indicated file. Allowed for use with -report.")
    parser.add_argument(
        "-x",
        metavar="<xfile>",
        help="File with the names of numeric properties from <infile> that "
        "should be included in the pool of independent variables.")


def add_report_arguments(parser):
    """
    Options to generate a summary of all models or a detailed report for a
    single model.

    :param parser: to add report generation args to.
    :type parser: argparse.ArgumentParser
    """

    group = parser.add_argument_group('report generation arguments',
                                      'Options to specify report generation.')
    group.add_argument(
        "-report",
        action="store_true",
        help="Write a summary of all models or generate a detailed report for "
        "a single model.")


def add_jobcontrol_arguments(parser):
    """
    Options to manipulate job control related parameters.

    :param parser: to add job controls args to.
    :type parser: argparse.ArgumentParser
    """

    group = parser.add_argument_group(
        'jobcontrol arguments',
        'Options to manipulate job control related parameters.')
    group.add_argument(
        "-JOB", metavar="<jobname>", help="Override the default job name.")
    group.add_argument(
        "-HOST",
        metavar="<host>[:<n>]",
        help="Run job on <host>. Include :<n> to specify the maximum number "
        "of jobs to run at a time on host")
    group.add_argument(
        "-NUM_SUBJOBS",
        metavar="<number>",
        help="Number of subjobs to create. If this option is omitted, "
        "-NUM_SUBJOBS will be set to the number of cpus requested with "
        "the -HOST option.",
        type=int,
    )
    group.add_argument(
        "-TMPDIR", metavar="<dir>", help="Store temporary job files in <dir>.")
    group.add_argument(
        "-WAIT",
        action="store_true",
        help="Do not return prompt until the job completes.")
    group.add_argument(
        "-SUBJOB",
        action="store_true",
        help="Denotes whether this task is a subjob or not")


def validate_args(args):
    """
    Validates command-line specified arguments.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :return: tuple of validation success, and error message
    :rtype: bool, str
    """

    if args.build and args.regression == args.classification:
        return (False,
                "Build must be either in Classification or Regression mode")
    if args.build + args.predict + args.report != 1:
        return False, "-build, -predict, and -report are mutually exclusive"

    # Confirm all given file paths exist
    given_files = [f for f in (args.i, args.x) if f]
    given_files_exist = [os.path.exists(f) for f in given_files]
    if not all(given_files_exist):
        missing_file = given_files[given_files_exist.index(False)]
        return False, "File not found: " + missing_file

    # Confirm input and holdout / predict output file formats match
    if args.build and args.holdout and not same_file_type(
        [args.i, args.holdout]):
        return (False, "FileType Mismatch: Input file and holdout file types "
                "must match.")
    if args.predict and not same_file_type([args.i, args.pred]):
        return (False, "FileType Mismatch: Input file and prediction output "
                "file types must match.")

    return True, ""


def parse_args(argv=None):
    """
    Parses and validates command-line arguments.

    :param argv: argument list
    :type argv: list(str)

    :return: argument namespace with command line options
    :rtype: argparse.Namespace
    """

    parser = get_parser()
    args = parser.parse_args(argv)

    validated, msg = validate_args(args)
    if not validated:
        parser.error(msg)

    return args


# =============================================================================
# Job spec & JobNames
# =============================================================================


def get_job_spec_from_args(argv):
    """
    Return a JobSpecification necessary to run this script on a remote
    machine (e.g. under job control with the launch.py script).

    :param argv: list of command-line arguments, including the script name at
        [0], matching: $SCHRODINGER/run __file__ sys.argv
    :type argv: list(str)

    :return: Serializable job definition
    :rtype: launchapi.JobSpecification
    """

    # Parse and validate arguments - first argument is this script
    args = parse_args(argv[1:])

    # Create a job specification and register input / output / log files
    job_builder = launchapi.JobSpecificationArgsBuilder(
        argv, use_jobname_log=True)
    job_builder.setProgramName(LIGAND_ML_DRIVER)

    # Set jobname depending on task type
    jobname = get_jobname(args)
    job_builder.setJobname(jobname)

    # Register input files
    if args.predict or args.report:
        job_builder.setInputFile(args.model)
    if args.i:
        job_builder.setInputFile(args.i)
    if args.holdout:
        job_builder.setInputFile(args.holdout)
    if args.x:
        job_builder.setInputFile(args.x)

    # Register output files
    if args.build:
        job_builder.setOutputFile(args.model)
    if args.pred:
        job_builder.setOutputFile(args.pred, incorporate=True)
    if args.o:
        job_builder.setOutputFile(args.o)
    job_builder.setOutputFile(jobname + '.okay')
    job_builder.setOutputFile(jobname + '.errors')

    return job_builder.getJobSpec()


def get_jobname(args):
    """
    Determine the job name depending on the task set by the command-line args.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :return: job name
    :rtype: str
    """

    if args.JOB:
        return args.JOB
    basename = fileutils.get_basename(args.model)
    if args.build:
        return basename + '_build'
    elif args.predict:
        return basename + '_predict'
    elif args.report:
        return basename + '_report'
    raise ValueError("-build, -predict, and -report are mutually exclusive")


def get_subjob_names(num_subjobs, prefix):
    """
    Returns a list of subjobs names of the form <prefix>_sub_<n>, where <n>
    runs from 1 to num_subjobs.

    :param num_subjobs: Number of subjobs
    :type num_subjobs: int

    :param prefix: Prefix for all subjob names
    :type prefix: str

    :return: Subjob names
    :rtype: list(str)
    """

    subjob_names = []
    for i in range(num_subjobs):
        subjob_names.append("%s_sub_%d" % (prefix, i + 1))

    return subjob_names


def get_model_subjob_names(num_subjobs, model):
    """
    Returns a list of model names of the form <model_prefix>_sub_<n>, where <n>
    runs from 1 to num_subjobs.

    :param num_subjobs: Number of subjobs
    :type num_subjobs: int

    :param model: Model name
    :type model: str

    :return: Model names
    :rtype: list(str)
    """

    prefix, ext = os.path.splitext(model)
    model_names = get_subjob_names(num_subjobs, prefix)
    return [m + ext for m in model_names]


# =============================================================================
# Command line args helper functions
# =============================================================================


def get_subjob_pred_argv(args):
    """
        Returns a command containing arguments that are common to all
        prediction subjobs

        :param args: argument namespace with command line options
        :type args: argparse.Namespace

        :return: command with common arguments
        :rtype: list[str]
        """

    exclude_args = ('model', 'JOB', 'HOST', 'NUM_SUBJOBS', 'i', 'pred')
    return get_argv_from_namespace(args, exclude_args)


def get_subjob_argv(args):
    """
    Returns a command containing arguments that are common to all subjobs.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :return: command with common arguments
    :rtype: list[str]
    """

    exclude_args = ('model', 'JOB', 'HOST', 'NUM_SUBJOBS')
    return get_argv_from_namespace(args, exclude_args)


def get_argv_from_namespace(args, exclude_args=None):
    """
    Generates list of arguments from an argparse.Namespace object.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :param exclude_args: List of arguments to exclude from the generated `argv`.
    :type exclude_args: iter(str)

    :return: list of arguments
    :rtype: list[str]
    """

    argv = []
    exclude_args = exclude_args or []
    for key, value in args.__dict__.items():
        if key in exclude_args:
            continue  # ignore excluded args
        flag = '-' + key
        if value is not None:
            if value is True:
                argv.append(flag)  # boolean flag
            elif value is not False:
                argv.extend([flag, str(value)])  # key=value flag
    return argv


def get_dict_from_namespace(args, exclude_args=None):
    """
    Generates dict of arguments from an argparse.Namespace object with the
    requested args removed.

    :param args: argument namespace with command line options
    :type args: argparse.Namespace

    :param exclude_args: List of arguments to exclude from the generated `argv`.
    :type exclude_args: iter(str)

    :return: dict of arguments
    :rtype: dict(str, Any)
    """

    argd = copy.deepcopy(args.__dict__)
    exclude_args = exclude_args or []
    for key in exclude_args:
        argd.pop(key, None)
    return argd


def same_file_type(files):
    """
    Check if all given files have the same file extension. Files without an
    extension do not get a free pass.

    :param files: Iterable of file names
    :type files: iter(str)

    :return: whether all file names hold the same extension.
    :rtype: bool
    """
    file_types = [fileutils.get_structure_file_format(f) for f in files]
    return len(set(file_types)) == 1


# =============================================================================
# Report functions
# =============================================================================


def report(args):
    build_args_dict = unzip_files(args.model)
    build_folder = os.path.splitext(build_args_dict['model'])[0]

    # smasher report.json
    report_fp = os.path.join(build_folder, 'smasher', 'report.json')
    if not os.path.exists(report_fp):
        raise ValueError('No smasher model in the model archive.')
    with open(report_fp, 'r') as fh:
        report_json = json.load(fh)

    # update plot path
    plot_path = os.path.basename(report_json['metrics'][-1]['plot_path'])
    plot_fn = plot_path + '.png'
    if build_args_dict['classification']:
        plot_fn = plot_path + '-roc_auc.png'
    report_json['plot_path'] = os.path.join(build_folder, 'smasher', 'metrics',
                                            plot_fn)
    # remove old plot path from metrics
    for m in report_json['metrics']:
        m.pop('plot_path')

    # custom descriptors
    report_json['custom_descriptors'] = build_args_dict['descriptors']

    # additional info
    report_json['Mode'] = ('Classification' if build_args_dict['classification']
                           else 'Regression')
    return report_json


def unzip_files(model_path):
    zfile = zipfile.ZipFile(model_path, 'r')
    zfile.extractall()
    f_args_data = None
    for filename in zfile.namelist():
        if filename.endswith('job_args.json'):
            f_args_data = zfile.open(filename).read()
    if not f_args_data:
        raise ValueError('No job_args.json file in the model archive')
    build_args_dict = json.loads(f_args_data)
    return build_args_dict


# =============================================================================
# Main
# =============================================================================


@driver_decorator.main_wrapper(LIGAND_ML_DRIVER)
def main():

    # Parse and validate arguments
    args = parse_args()

    # Determine the number of subjobs to distribute over from -HOST argument
    n_jobs = args.NUM_SUBJOBS
    if n_jobs is None:
        host_list = jobcontrol.get_backend_host_list()
        n_jobs = jobcontrol.calculate_njobs(host_list)

    # Run serial or distributed depending on number of hosts specified
    if n_jobs > 1 and args.build and not args.SUBJOB:
        run_distributed(args, n_jobs)
    elif n_jobs > 1 and args.predict and not args.SUBJOB:
        args.NUM_SUBJOBS = n_jobs
        check_prediction_split(args)
    else:
        run_serial(args)

    logger.info('LigandML driver completed successfully')


if __name__ == "__main__":
    main()
