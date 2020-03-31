import os
import pytest
import numpy as np
import behavenet.fitting.ae_model_architecture_generator as utils


def test_draw_archs():

    n_archs = 3
    n_ae_latents = 6

    # no check memory
    archs = utils.draw_archs(
        batch_size=100, input_dim=[2, 32, 32], n_ae_latents=n_ae_latents, n_archs=n_archs,
        check_memory=False, mem_limit_gb=None)
    assert len(archs) == n_archs
    for arch1 in archs:
        assert arch1['n_ae_latents'] == n_ae_latents
        matching = 0
        for arch2 in archs:
            if arch1 == arch2:
                matching += 1
        assert matching == 1

    # check memory
    mem_limit_gb = 1
    archs = utils.draw_archs(
        batch_size=100, input_dim=[2, 32, 32], n_ae_latents=n_ae_latents, n_archs=n_archs,
        check_memory=True, mem_limit_gb=mem_limit_gb)
    assert len(archs) == n_archs
    for arch1 in archs:
        assert arch1['n_ae_latents'] == n_ae_latents
        assert arch1['mem_size_gb'] < mem_limit_gb
        matching = 0
        for arch2 in archs:
            if arch1 == arch2:
                matching += 1
        assert matching == 1


def test_get_possible_arch():

    input_dim = [2, 32, 32]
    arch_seed = 0

    # proper functioning
    n_ae_latents = 6
    arch = utils.get_possible_arch(input_dim, n_ae_latents, arch_seed)
    assert arch['n_ae_latents'] == n_ae_latents

    # raise exception if too many latents (max = 64)
    n_ae_latents = 65
    with pytest.raises(ValueError):
        utils.get_possible_arch(input_dim, n_ae_latents, arch_seed)


def test_get_encoding_block_conv():

    input_dim = [2, 32, 32]
    n_ae_latents = 6

    # possible options for the architecture
    opts = {}
    opts['possible_kernel_sizes'] = np.asarray([3, 5])
    opts['possible_strides'] = np.asarray([1, 2])
    opts['possible_strides_probs'] = np.asarray([0.1, 0.9])
    opts['possible_max_pool_sizes'] = np.asarray([2])
    opts['possible_n_channels'] = np.asarray([16, 32, 64, 128])
    opts['prob_stopping'] = np.arange(0, 1, .05)
    opts['max_latents'] = 64

    arch = {}
    arch['ae_input_dim'] = input_dim
    arch['model_type'] = 'conv'
    arch['n_ae_latents'] = n_ae_latents
    arch['ae_decoding_last_FF_layer'] = 0
    arch['ae_network_type'] = 'strides_only'
    arch['ae_padding_type'] = 'valid'

    # using correct options (all convolutional)
    np.random.seed(4)
    arch = utils.get_encoding_conv_block(arch, opts)
    for i in range(len(arch['ae_encoding_n_channels'])):
        assert arch['ae_encoding_layer_type'][i] in ['conv']
        assert arch['ae_encoding_n_channels'][i] in opts['possible_n_channels']
        assert arch['ae_encoding_kernel_size'][i] in opts['possible_kernel_sizes']
        assert arch['ae_encoding_stride_size'][i] in opts['possible_strides']

    # usng correct options (with maxpool)
    np.random.seed(6)
    arch['ae_network_type'] = 'max_pooling'
    arch = utils.get_encoding_conv_block(arch, opts)
    for i in range(len(arch['ae_encoding_n_channels'])):
        assert arch['ae_encoding_layer_type'][i] in ['conv', 'maxpool']
        assert arch['ae_encoding_n_channels'][i] in opts['possible_n_channels']
        if arch['ae_encoding_layer_type'][i] == 'conv':
            assert arch['ae_encoding_kernel_size'][i] in opts['possible_kernel_sizes']
            assert arch['ae_encoding_stride_size'][i] in opts['possible_strides']
        else:
            assert arch['ae_encoding_kernel_size'][i] in opts['possible_max_pool_sizes']
            assert arch['ae_encoding_stride_size'][i] in opts['possible_max_pool_sizes']


def test_get_decoding_conv_block():

    input_dim = [2, 128, 128]
    n_ae_latents = 6

    # possible options for the architecture
    opts = {}
    opts['possible_kernel_sizes'] = np.asarray([3, 5])
    opts['possible_strides'] = np.asarray([1, 2])
    opts['possible_strides_probs'] = np.asarray([0.1, 0.9])
    opts['possible_max_pool_sizes'] = np.asarray([2])
    opts['possible_n_channels'] = np.asarray([16, 32, 64, 128])
    opts['prob_stopping'] = np.arange(0, 1, .05)
    opts['max_latents'] = 64

    arch = {}
    arch['ae_input_dim'] = input_dim
    arch['model_type'] = 'conv'
    arch['n_ae_latents'] = n_ae_latents
    arch['ae_decoding_last_FF_layer'] = 0
    arch['ae_network_type'] = 'strides_only'
    arch['ae_padding_type'] = 'valid'

    # using correct options (all convolutional)
    np.random.seed(16)
    arch = utils.get_encoding_conv_block(arch, opts)
    arch = utils.get_decoding_conv_block(arch)
    assert arch['ae_decoding_n_channels'][-1] == input_dim[0]
    for i in range(len(arch['ae_decoding_n_channels']) - 1):
        assert arch['ae_decoding_layer_type'][i] in ['convtranspose']
        assert arch['ae_decoding_n_channels'][i] == arch['ae_encoding_n_channels'][-2-i]
        assert arch['ae_decoding_kernel_size'][i] == arch['ae_encoding_kernel_size'][-1-i]
        assert arch['ae_decoding_stride_size'][i] == arch['ae_encoding_stride_size'][-1-i]

    # using correct options (with maxpool)
    np.random.seed(16)
    arch['ae_network_type'] = 'max_pooling'
    arch = utils.get_encoding_conv_block(arch, opts)
    arch = utils.get_decoding_conv_block(arch)
    print(arch)
    for i in range(len(arch['ae_decoding_n_channels']) - 1):
        assert arch['ae_decoding_layer_type'][i] in ['convtranspose', 'unpool']
        assert arch['ae_decoding_n_channels'][i] == arch['ae_encoding_n_channels'][-2-i]
        assert arch['ae_decoding_kernel_size'][i] == arch['ae_encoding_kernel_size'][-1-i]
        assert arch['ae_decoding_stride_size'][i] == arch['ae_encoding_stride_size'][-1-i]

    # using correct options (with final ff layer)
    arch['ae_decoding_last_FF_layer'] = True
    arch = utils.get_decoding_conv_block(arch)
    assert arch['ae_decoding_n_channels'][-1] == 16


def test_calculate_output_dim():

    # try all even/odd combos for input_dim/kernel/stride

    # ----------------------
    # conv layers - same
    # ----------------------
    input_dim, kernel, stride = 16, 4, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 6
    assert before == 1
    assert after == 2

    input_dim, kernel, stride = 17, 4, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 6
    assert before == 1
    assert after == 1

    input_dim, kernel, stride = 16, 3, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 6
    assert before == 1
    assert after == 1

    input_dim, kernel, stride = 17, 3, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 6
    assert before == 0
    assert after == 1

    input_dim, kernel, stride = 16, 4, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 8
    assert before == 1
    assert after == 1

    input_dim, kernel, stride = 17, 4, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 9
    assert before == 1
    assert after == 2

    input_dim, kernel, stride = 16, 3, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 8
    assert before == 0
    assert after == 1

    input_dim, kernel, stride = 17, 3, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'conv')
    assert out == 9
    assert before == 1
    assert after == 1

    # ----------------------
    # conv layers - valid
    # ----------------------
    input_dim, kernel, stride = 16, 4, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 5
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 4, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 5
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 16, 3, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 5
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 3, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 5
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 16, 4, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 7
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 4, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 7
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 16, 3, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 7
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 3, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'conv')
    assert out == 8
    assert before == 0
    assert after == 0

    # ----------------------
    # conv layers - other
    # ----------------------
    with pytest.raises(NotImplementedError):
        utils.calculate_output_dim(input_dim, kernel, stride, 'test', 'conv')

    # ----------------------
    # maxpool layers - kern
    # ----------------------
    with pytest.raises(NotImplementedError):
        utils.calculate_output_dim(input_dim, 3, stride, 'test', 'conv')

    # ----------------------
    # maxpool layers - same
    # ----------------------
    input_dim, kernel, stride = 16, 2, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'maxpool')
    assert out == 6
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 2, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'maxpool')
    assert out == 6
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 16, 2, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'maxpool')
    assert out == 8
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 2, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'same', 'maxpool')
    assert out == 9
    assert before == 0
    assert after == 0

    # ----------------------
    # maxpool layers - valid
    # ----------------------
    input_dim, kernel, stride = 16, 2, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'maxpool')
    assert out == 5
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 2, 3
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'maxpool')
    assert out == 6
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 16, 2, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'maxpool')
    assert out == 8
    assert before == 0
    assert after == 0

    input_dim, kernel, stride = 17, 2, 2
    out, before, after = utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'maxpool')
    assert out == 8
    assert before == 0
    assert after == 0

    # ----------------------
    # maxpool layers - other
    # ----------------------
    with pytest.raises(NotImplementedError):
        utils.calculate_output_dim(input_dim, kernel, stride, 'test', 'maxpool')

    # ----------------------
    # other layers
    # ----------------------
    with pytest.raises(NotImplementedError):
        utils.calculate_output_dim(input_dim, kernel, stride, 'valid', 'test')


def test_estimate_model_footprint():

    from behavenet.models.aes import AE

    n_archs = 3
    input_dim = [2, 128, 128]
    n_ae_latents = 12
    archs = utils.draw_archs(
        batch_size=100, input_dim=input_dim, n_ae_latents=n_ae_latents, n_archs=n_archs,
        check_memory=False, mem_limit_gb=20)
    for arch in archs:
        arch['model_class'] = 'ae'
        arch['n_input_channels'] = input_dim[0]
        arch['y_pixels'] = input_dim[1]
        arch['x_pixels'] = input_dim[2]

    model = AE(archs[0])
    f0 = utils.estimate_model_footprint(model, tuple([100] + input_dim))
    assert 290 < f0 / 1e6 < 310

    model = AE(archs[1])
    f1 = utils.estimate_model_footprint(model, tuple([100] + input_dim))
    assert 600 < f1 / 1e6 < 650

    model = AE(archs[2])
    f2 = utils.estimate_model_footprint(model, tuple([100] + input_dim))
    assert 1000 < f2 / 1e6 < 1200

    # cutoff size
    f2a = utils.estimate_model_footprint(model, tuple([100] + input_dim), 0.1)
    assert 100 < f2a < f2


def test_get_handcrafted_dims():

    # symmetric arch
    arch0 = utils.load_default_arch()
    arch0['ae_input_dim'] = [2, 128, 128]
    arch0 = utils.get_handcrafted_dims(arch0, symmetric=True)
    assert arch0['ae_encoding_x_dim'] == [64, 32, 16, 8]
    assert arch0['ae_encoding_y_dim'] == [64, 32, 16, 8]
    assert arch0['ae_encoding_x_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert arch0['ae_encoding_y_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert arch0['ae_decoding_x_dim'] == [16, 32, 64, 128]
    assert arch0['ae_decoding_y_dim'] == [16, 32, 64, 128]
    assert arch0['ae_decoding_x_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert arch0['ae_decoding_y_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]

    # asymmetric arch (TODO: source code not updated)
    arch1 = utils.load_default_arch()
    arch1['ae_input_dim'] = [2, 128, 128]
    arch1['ae_decoding_n_channels'] = [64, 32, 32]
    arch1['ae_decoding_kernel_size'] = [5, 5, 5]
    arch1['ae_decoding_stride_size'] = [2, 2, 2]
    arch1['ae_decoding_layer_type'] = ['conv', 'conv', 'conv']
    arch1['ae_decoding_starting_dim'] = [1, 8, 8]
    arch1 = utils.get_handcrafted_dims(arch1, symmetric=False)
    assert arch1['ae_encoding_x_dim'] == [64, 32, 16, 8]
    assert arch1['ae_encoding_y_dim'] == [64, 32, 16, 8]
    assert arch1['ae_encoding_x_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert arch1['ae_encoding_y_padding'] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert arch1['ae_decoding_x_dim'] == [15, 29, 57]
    assert arch1['ae_decoding_y_dim'] == [15, 29, 57]
    assert arch1['ae_decoding_x_padding'] == [(2, 2), (2, 2), (2, 2)]
    assert arch1['ae_decoding_y_padding'] == [(2, 2), (2, 2), (2, 2)]

    # raise exception if asymmetric arch and max pooling
    arch2 = utils.load_default_arch()
    arch2['ae_input_dim'] = [2, 128, 128]
    arch2['ae_network_type'] = 'max_pooling'
    with pytest.raises(NotImplementedError):
        utils.get_handcrafted_dims(arch2, symmetric=False)


def test_load_handcrafted_arch():

    input_dim = [2, 128, 128]
    n_ae_latents = 12

    # use default arch
    ae_arch_json = None
    arch = utils.load_handcrafted_arch(input_dim, n_ae_latents, ae_arch_json, check_memory=False)
    assert arch['n_input_channels'] == input_dim[0]
    assert arch['y_pixels'] == input_dim[1]
    assert arch['x_pixels'] == input_dim[2]
    assert arch['ae_input_dim'] == input_dim
    assert arch['n_ae_latents'] == n_ae_latents
    assert arch['ae_encoding_n_channels'] == [32, 64, 256, 512]

    # load arch from json
    ae_arch_json = os.path.join(
        os.getcwd(), 'behavenet', 'json_configs', 'ae_jsons', 'ae_arch_2.json')
    arch = utils.load_handcrafted_arch(input_dim, n_ae_latents, ae_arch_json, check_memory=False)
    assert arch['n_input_channels'] == input_dim[0]
    assert arch['y_pixels'] == input_dim[1]
    assert arch['x_pixels'] == input_dim[2]
    assert arch['ae_input_dim'] == input_dim
    assert arch['n_ae_latents'] == n_ae_latents
    assert arch['ae_encoding_n_channels'] == [64, 64, 64, 64, 64]

    # use default arch when json does not exist
    ae_arch_json = os.path.join(
        os.getcwd(), 'behavenet', 'json_configs', 'ae_jsons', 'ae_arch_3.json')
    arch = utils.load_handcrafted_arch(input_dim, n_ae_latents, ae_arch_json, check_memory=False)
    assert arch['n_input_channels'] == input_dim[0]
    assert arch['y_pixels'] == input_dim[1]
    assert arch['x_pixels'] == input_dim[2]
    assert arch['ae_input_dim'] == input_dim
    assert arch['n_ae_latents'] == n_ae_latents
    assert arch['ae_encoding_n_channels'] == [32, 64, 256, 512]

    # check memory runs
    ae_arch_json = None
    arch = utils.load_handcrafted_arch(
        input_dim, n_ae_latents, ae_arch_json, check_memory=True, batch_size=10, mem_limit_gb=20)
    assert arch['n_input_channels'] == input_dim[0]
    assert arch['y_pixels'] == input_dim[1]
    assert arch['x_pixels'] == input_dim[2]
    assert arch['ae_input_dim'] == input_dim
    assert arch['n_ae_latents'] == n_ae_latents
    assert arch['ae_encoding_n_channels'] == [32, 64, 256, 512]

    # raise exception when not enough gpu memory
    ae_arch_json = None
    with pytest.raises(ValueError):
        utils.load_handcrafted_arch(
            input_dim, n_ae_latents, ae_arch_json,
            check_memory=True, batch_size=10, mem_limit_gb=0.1)


def test_load_default_arch():

    required_keys = [
        'ae_network_type',
        'ae_padding_type',
        'ae_batch_norm',
        'ae_batch_norm_momentum',
        'symmetric_arch',
        'ae_encoding_n_channels',
        'ae_encoding_kernel_size',
        'ae_encoding_stride_size',
        'ae_encoding_layer_type',
        'ae_decoding_last_FF_layer']
    arch = utils.load_default_arch()
    returned_keys = list(arch.keys())
    for key in required_keys:
        assert key in returned_keys
