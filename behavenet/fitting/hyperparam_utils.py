import commentjson
from test_tube import HyperOptArgumentParser
from behavenet import get_user_dir
from behavenet.fitting.ae_model_architecture_generator import draw_handcrafted_archs
import sys

def get_all_params(search_type='grid_search', args=None):

    # Raise error if user has other command line arguments specified (as could override configs in confusing ways)
    if args is not None and len(args) != 8:
        raise ValueError('No command line arguments allowed other than config file names')
    elif len(sys.argv[1:]) != 8:
        raise ValueError('No command line arguments allowed other than config file names')

    # Create parser
    parser = HyperOptArgumentParser(strategy=search_type)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--training_config', type=str)
    parser.add_argument('--compute_config', type=str)

    namespace, extra = parser.parse_known_args(args)

    # Add arguments from all configs
    configs = [namespace.data_config, namespace.model_config, namespace.training_config, namespace.compute_config]
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
    if isinstance(value, list):
        parser.opt_list('--' + arg_name, options=value, tunable=True)
    else:
        parser.add_argument('--' + arg_name, default=value)


def add_dependent_params(parser, namespace):
    if namespace.model_class == 'ae':

        if namespace.arch_types == "default":

            which_handcrafted_archs = 0

            list_of_archs = draw_handcrafted_archs(
                [namespace.n_input_channels, namespace.y_pixels, namespace.x_pixels],
                namespace.n_ae_latents,
                which_handcrafted_archs,
                check_memory=True,
                batch_size=namespace.approx_batch_size,
                mem_limit_gb=namespace.mem_limit_gb)

            parser.opt_list('--architecture_params', options=[list_of_archs[0]], tunable=True)
            parser.add_argument('--max_latents', default=64)

        else:
            raise NotImplementedError('Other architectures not specified')



