import commentjson
import datetime
import sys
import os
from subprocess import call
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster, AbstractCluster
from behavenet import get_user_dir
from behavenet.models.ae_model_architecture_generator import load_handcrafted_arches


def get_all_params(search_type='grid_search', args=None):

    # Raise error if user has other command line arguments specified (as could override configs in
    # confusing ways)
    if args is not None and len(args) != 8:
        raise ValueError('No command line arguments allowed other than config file names')
    elif args is None and len(sys.argv[1:]) != 8:
        raise ValueError('No command line arguments allowed other than config file names')

    # Create parser
    parser = HyperOptArgumentParser(strategy=search_type)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--training_config', type=str)
    parser.add_argument('--compute_config', type=str)

    namespace, extra = parser.parse_known_args(args)

    # Add arguments from all configs
    configs = [
        namespace.data_config,
        namespace.model_config,
        namespace.training_config,
        namespace.compute_config]
    for config in configs:
        config_json = commentjson.load(open(config, 'r'))
        for (key, value) in config_json.items():
            add_to_parser(parser, key, value)

    # Add save/user dirs
    parser.add_argument('--save_dir', default=get_user_dir('save'), type=str)
    parser.add_argument('--data_dir', default=get_user_dir('data'), type=str)

    # Add parameters dependent on previous inputs
    namespace, extra = parser.parse_known_args(args)
    add_dependent_params(parser, namespace)

    return parser.parse_args(args)


def add_to_parser(parser, arg_name, value):
    if arg_name == 'n_ae_latents':
        parser.add_argument('--' + 'n_latents', default=str(value))
    else:
        if isinstance(value, list):
            parser.opt_list('--' + arg_name, options=value, tunable=True)
        else:
            parser.add_argument('--' + arg_name, default=value)


def add_dependent_params(parser, namespace):
    """Add params that are derived from json arguments."""

    if namespace.model_class == 'ae' \
            or namespace.model_class == 'vae' \
            or namespace.model_class == 'beta-tcvae' \
            or namespace.model_class == 'cond-vae' \
            or namespace.model_class == 'cond-ae' \
            or namespace.model_class == 'cond-ae-msp' \
            or namespace.model_class == 'ps-vae' \
            or namespace.model_class == 'msps-vae' \
            or namespace.model_class == 'labels-images':

        if namespace.model_type == 'conv':
            max_latents = 64
            parser.add_argument('--max_latents', default=max_latents)
            arch_dicts = load_handcrafted_arches(
                [namespace.n_input_channels, namespace.y_pixels, namespace.x_pixels],
                namespace.n_latents,
                namespace.ae_arch_json,
                check_memory=False,
                batch_size=namespace.approx_batch_size,
                mem_limit_gb=namespace.mem_limit_gb)
            parser.opt_list('--architecture_params', options=arch_dicts, tunable=True)

        elif namespace.model_type == 'linear':
            parser.add_argument('--n_ae_latents', default=namespace.n_latents, type=int)

        else:
            raise ValueError('%s is not a valid model type' % namespace.model_type)

        # for i, arch_dict in enumerate(arch_dicts):
        #     if (arch_dict['ae_encoding_n_channels'][-1]
        #             * arch_dict['ae_encoding_x_dim'][-1]
        #             * arch_dict['ae_encoding_y_dim'][-1]) < namespace.n_latents[i]:
        #         raise ValueError('Bottleneck smaller than number of latents')

    else:
        if getattr(namespace, 'n_latents', False):
            parser.add_argument('--n_ae_latents', default=namespace.n_latents, type=int)

    if namespace.model_class.find('neural') > -1:

        # parse "subsample_idxs_names" arg to determine which index keys to fit; the code below
        # currently supports 'all' (all idx keys) or a single string (single idx key)
        if namespace.subsample_method != 'none':
            if namespace.subsample_idxs_dataset == 'all':
                from behavenet.data.utils import get_region_list
                idx_list = get_region_list(namespace)
                parser.opt_list(
                    '--subsample_idxs_name', options=idx_list, tunable=True)
            elif isinstance(namespace.subsample_idxs_dataset, str):
                parser.add_argument(
                    '--subsample_idxs_name', default=namespace.subsample_idxs_dataset)
            else:
                raise ValueError(
                    '%s is an invalid data type for "subsample_idxs_dataset" key in data json; ' %
                    type(namespace.subsample_idxs_dataset) +
                    'must be a string ("all" or "name")')
    else:
        pass


class CustomSlurmCluster(SlurmCluster):

    def __init__(self, master_slurm_file, *args, **kwargs):
        super(CustomSlurmCluster, self).__init__(*args, **kwargs)

        self.master_slurm_file = master_slurm_file

    def schedule_experiment(self, trial_params, exp_i):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        timestamp = 'trial_{}_{}'.format(exp_i, timestamp)

        # generate command
        slurm_cmd_script_path = os.path.join(
            self.slurm_files_log_path, '{}_slurm_cmd.sh'.format(timestamp))
        run_cmd = self.__get_run_command(
            trial_params, slurm_cmd_script_path, timestamp, exp_i, self.on_gpu)
        sbatch_params = open(self.master_slurm_file, 'r').read()
        slurm_cmd = sbatch_params+run_cmd
        self._SlurmCluster__save_slurm_cmd(slurm_cmd, slurm_cmd_script_path)

        # run script to launch job
        print('\nlaunching exp...')
        result = call('{} {}'.format(AbstractCluster.RUN_CMD, slurm_cmd_script_path), shell=True)
        if result == 0:
            print('launched exp ', slurm_cmd_script_path)
        else:
            print('launch failed...')

    def __get_run_command(self, trial, slurm_cmd_script_path, timestamp, exp_i, on_gpu):
        trial_args = self._SlurmCluster__get_hopt_params(trial)
        trial_args = '{} --{} {} --{} {}'.format(trial_args,
                                                 HyperOptArgumentParser.SLURM_CMD_PATH,
                                                 slurm_cmd_script_path,
                                                 HyperOptArgumentParser.SLURM_EXP_CMD,
                                                 exp_i)

        cmd = 'srun {} {} {}'.format(self.python_cmd, self.script_name, trial_args)
        return cmd


def get_slurm_params(hyperparams):

    cluster = CustomSlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.slurm_log_path,
        python_cmd='python3',
        master_slurm_file=hyperparams.slurm_param_file,
    )

    return cluster
