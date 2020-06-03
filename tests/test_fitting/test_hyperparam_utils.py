import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # get rid of tensorboard warnings
import os
import sys
import h5py
import pytest
import numpy as np
from test_tube import HyperOptArgumentParser
from behavenet.fitting import hyperparam_utils as utils


def test_get_all_params():

    # not enough args
    args = [
        '--data_config', 'data.json',
        '--model_config', 'model.json',
        '--training_config', 'training.json']
    with pytest.raises(ValueError):
        utils.get_all_params(search_type='grid_search', args=args)

    # too many args
    args = [
        '--data_config', 'data.json',
        '--model_config', 'model.json',
        '--training_config', 'training.json',
        '--compute_config', 'compute.json',
        '--model_class', 'ae']
    with pytest.raises(ValueError):
        utils.get_all_params(search_type='grid_search', args=args)

    # correct args, substituted into sys.argv
    data_config = os.path.join(
        os.getcwd(), 'behavenet/json_configs/data_default.json')
    model_config = os.path.join(
        os.getcwd(), 'behavenet/json_configs/arhmm_jsons/arhmm_model.json')
    training_config = os.path.join(
        os.getcwd(), 'behavenet/json_configs/arhmm_jsons/arhmm_training.json')
    compute_config = os.path.join(
            os.getcwd(), 'behavenet/json_configs/arhmm_jsons/arhmm_compute.json')
    args = [
        '--data_config', data_config,
        '--model_config', model_config,
        '--training_config', training_config,
        '--compute_config', compute_config]
    old_sys_argv = sys.argv
    sys.argv = [old_sys_argv[0]] + args
    parser = utils.get_all_params(search_type='grid_search')
    # test a couple args
    assert parser.data_config == data_config
    assert parser.model_config == model_config
    assert parser.training_config == training_config
    assert parser.compute_config == compute_config


def test_add_to_parser():

    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'test0', '4')
    utils.add_to_parser(parser, 'test1', '5')
    utils.add_to_parser(parser, 'test2', [1, 2, 3])

    namespace, _ = parser.parse_known_args(['--test0', '3'])

    # single argument
    assert namespace.test0 == '3'  # user defined arg
    assert namespace.test1 == '5'  # default arg

    # list argument
    assert namespace.test2 is None
    assert parser.opt_args['--test2'].opt_values == [1, 2, 3]
    assert parser.opt_args['--test2'].tunable


def test_add_dependent_params(tmpdir):

    # -----------------
    # ae
    # -----------------
    # arch params correctly added to parser
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'model_class', 'ae')
    utils.add_to_parser(parser, 'n_ae_latents', 32)
    utils.add_to_parser(parser, 'n_input_channels', 2)
    utils.add_to_parser(parser, 'y_pixels', 32)
    utils.add_to_parser(parser, 'x_pixels', 32)
    utils.add_to_parser(parser, 'ae_arch_json', None)
    utils.add_to_parser(parser, 'approx_batch_size', 200)
    utils.add_to_parser(parser, 'mem_limit_gb', 10)
    namespace, _ = parser.parse_known_args([])
    utils.add_dependent_params(parser, namespace)
    assert '--architecture_params' in parser.opt_args

    # raise exception when max latents exceeded
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'model_class', 'ae')
    utils.add_to_parser(parser, 'n_ae_latents', 100000)
    utils.add_to_parser(parser, 'n_input_channels', 2)
    utils.add_to_parser(parser, 'y_pixels', 32)
    utils.add_to_parser(parser, 'x_pixels', 32)
    utils.add_to_parser(parser, 'ae_arch_json', None)
    utils.add_to_parser(parser, 'approx_batch_size', 200)
    utils.add_to_parser(parser, 'mem_limit_gb', 10)
    namespace, _ = parser.parse_known_args([])
    with pytest.raises(ValueError):
        utils.add_dependent_params(parser, namespace)

    # -----------------
    # neural
    # -----------------
    # make tmp hdf5 file
    path = tmpdir.join('data.hdf5')
    idx_data = {
        'i0': np.array([0, 1, 2]),
        'i1': np.array([3, 4, 5]),
        'i2': np.array([6, 7, 8])}
    with h5py.File(path, 'w') as f:
        group0 = f.create_group('regions')
        groupa = f.create_group('neural')
        group1 = group0.create_group('indxs')
        group1.create_dataset('i0', data=idx_data['i0'])
        group1.create_dataset('i1', data=idx_data['i1'])
        group1.create_dataset('i2', data=idx_data['i2'])

    # subsample idxs not added to parser when not requested
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'model_class', 'neural-ae')
    utils.add_to_parser(parser, 'subsample_method', 'none')
    namespace, _ = parser.parse_known_args([])
    utils.add_dependent_params(parser, namespace)
    assert '--subsample_idxs_name' not in parser.opt_args

    # subsample idxs added to parser when requested (all datasets)
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'data_dir', tmpdir)
    utils.add_to_parser(parser, 'lab', '')
    utils.add_to_parser(parser, 'expt', '')
    utils.add_to_parser(parser, 'animal', '')
    utils.add_to_parser(parser, 'session', '')
    utils.add_to_parser(parser, 'model_class', 'neural-ae')
    utils.add_to_parser(parser, 'subsample_method', 'single')
    utils.add_to_parser(parser, 'subsample_idxs_dataset', 'all')
    namespace, _ = parser.parse_known_args([])
    utils.add_dependent_params(parser, namespace)
    assert '--subsample_idxs_name' in parser.opt_args
    parser_vals = parser.opt_args['--subsample_idxs_name'].opt_values.keys()
    assert sorted(['i0', 'i1', 'i2']) == sorted(parser_vals)

    # subsample idxs added to parser when requested (single dataset)
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'data_dir', tmpdir)
    utils.add_to_parser(parser, 'lab', '')
    utils.add_to_parser(parser, 'expt', '')
    utils.add_to_parser(parser, 'animal', '')
    utils.add_to_parser(parser, 'session', '')
    utils.add_to_parser(parser, 'model_class', 'neural-ae')
    utils.add_to_parser(parser, 'subsample_method', 'single')
    utils.add_to_parser(parser, 'subsample_idxs_dataset', 'i0')
    namespace, _ = parser.parse_known_args([])
    utils.add_dependent_params(parser, namespace)
    parser.parse_args([])
    assert parser.parsed_args['subsample_idxs_name'] == 'i0'

    # raise exception when dataset is not a string
    parser = HyperOptArgumentParser(strategy='grid_search')
    utils.add_to_parser(parser, 'model_class', 'neural-ae')
    utils.add_to_parser(parser, 'subsample_method', 'single')
    utils.add_to_parser(parser, 'subsample_idxs_dataset', ['i0', 'i1'])
    namespace, _ = parser.parse_known_args([])
    with pytest.raises(ValueError):
        utils.add_dependent_params(parser, namespace)
