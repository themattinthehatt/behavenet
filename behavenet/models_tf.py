import torch
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from fitting.ae_model_architecture_generator import calculate_output_dim


class ConvAEEncoder(object):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams
        self.build_model()

    def build_model(self):

        self.encoder = []

        # Loop over layers (each conv/batch norm/max pool/relu chunk counts as
        # one layer for global_layer_num)
        global_layer_num = 0
        for i_layer in range(len(self.hparams['ae_encoding_n_channels'])):

            # only add if conv layer
            # (checks within this for next max pool layer)
            if self.hparams['ae_encoding_layer_type'][i_layer] == 'conv':

                # Convolution layer
                if i_layer == 0:
                    input_shape = (
                        self.hparams['ae_input_dim'][1],
                        self.hparams['ae_input_dim'][2],
                        self.hparams['ae_input_dim'][0])
                else:
                    input_shape = (
                        self.hparams['ae_encoding_y_dim'][i_layer - 1],
                        self.hparams['ae_encoding_x_dim'][i_layer - 1],
                        self.hparams['ae_encoding_n_channels'][i_layer - 1])
                self.encoder.append(keras.layers.Conv2D(
                    name='conv' + str(global_layer_num),
                    data_format='channels_last',
                    input_shape=input_shape,
                    filters=self.hparams['ae_encoding_n_channels'][i_layer],
                    kernel_size=self.hparams['ae_encoding_kernel_size'][i_layer],
                    strides=self.hparams['ae_encoding_stride_size'][i_layer],
                    padding='same'))

                # batch norm layer
                if self.hparams['ae_batch_norm']:
                    self.encoder.append(keras.layers.BatchNormalization(
                        name='bn' + str(global_layer_num),
                        axis=3,
                        momentum=self.hparams['ae_batch_norm_momentum']))

                # nonlinearity
                self.encoder.append(keras.layers.Activation(
                    tf.nn.leaky_relu,
                    name='relu' + str(global_layer_num)))

                global_layer_num += 1

        # Final FF layer to latents
        output_dim_x, _, _ = calculate_output_dim(
            self.hparams['ae_encoding_x_dim'][i_layer],
            self.hparams['ae_encoding_kernel_size'][i_layer],
            self.hparams['ae_encoding_stride_size'][i_layer],
            'same', 'conv')
        output_dim_y, _, _ = calculate_output_dim(
            self.hparams['ae_encoding_x_dim'][i_layer],
            self.hparams['ae_encoding_kernel_size'][i_layer],
            self.hparams['ae_encoding_stride_size'][i_layer],
            'same', 'conv')
        output_shape = (
            output_dim_y, output_dim_x,
            self.hparams['ae_encoding_n_channels'][i_layer])
        self.ff = keras.layers.Dense(
            units=self.hparams['n_ae_latents'],
            input_shape=(None, np.prod(output_shape)))

        # If VAE model, have additional FF layer to latent variances
        if self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')

    def forward(self, x):
        # x should be batch size x n channels x xdim x ydim

        # Loop over layers, have to collect pool_idx and output sizes if using max pooling to use in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            print(layer.name)
            # if isinstance(layer, nn.MaxPool2d):
            #     raise NotImplementedError
            #     # target_output_size.append(x.size())
            #     # x, idx = layer.apply(x)
            #     # pool_idx.append(idx)
            # else:
            #     x = layer.apply(x)
            x = layer.apply(x)

        if self.hparams['model_class'] == 'ae':
            self.x = self.ff.apply(tf.contrib.layers.flatten(x))
        elif self.hparams['model_class'] == 'vae':
            return NotImplementedError
        else:
            raise ValueError(
                self.hparams['model_class'] + ' not valid model type')

        return self.x


class ConvAEDecoder(object):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams
        self.build_model()

    def build_model(self):

        # First FF layer (from latents to size of last encoding layer)
        first_conv_size = np.prod(self.hparams['ae_decoding_starting_dim'])
        self.ff = keras.layers.Dense(
            units=first_conv_size,
            input_shape=(None, self.hparams['n_ae_latents']))

        self.decoder = []

        # Loop over layers (each unpool/convtranspose/batch norm/relu chunk
        # counts as one layer for global_layer_num)
        global_layer_num = 0

        for i_layer in range(len(self.hparams['ae_decoding_n_channels'])):

            if self.hparams['ae_decoding_layer_type'][i_layer] == 'convtranspose':

                # Unpooling layer
                if i_layer > 0 and self.hparams['ae_decoding_layer_type'][i_layer - 1] == 'unpool':
                    raise NotImplementedError
                    # self.decoder.add_module('maxunpool'+str(global_layer_num),nn.MaxUnpool2d(kernel_size=(int(self.hparams['ae_decoding_kernel_size'][i_layer-1]),int(self.hparams['ae_decoding_kernel_size'][i_layer-1])),stride = (int(self.hparams['ae_decoding_stride_size'][i_layer-1]),int(self.hparams['ae_decoding_stride_size'][i_layer-1])),padding=(self.hparams['ae_decoding_y_padding'][i_layer-1][0],self.hparams['ae_decoding_x_padding'][i_layer-1][0])))

                # ConvTranspose layer
                if i_layer == 0:
                    input_shape = (
                        self.hparams['ae_decoding_starting_dim'][1],
                        self.hparams['ae_decoding_starting_dim'][2],
                        self.hparams['ae_decoding_starting_dim'][0])
                else:
                    input_shape = (
                        self.hparams['ae_decoding_y_dim'][i_layer - 1],
                        self.hparams['ae_decoding_x_dim'][i_layer - 1],
                        self.hparams['ae_decoding_n_channels'][i_layer - 1])
                self.decoder.append(keras.layers.Conv2DTranspose(
                    name='convtranspose' + str(global_layer_num),
                    data_format='channels_last',
                    input_shape=input_shape,
                    filters=self.hparams['ae_decoding_n_channels'][i_layer],
                    kernel_size=self.hparams['ae_decoding_kernel_size'][i_layer],
                    strides=self.hparams['ae_decoding_stride_size'][i_layer],
                    padding='same'))

                # batch norm layer (but not just before output)
                if i_layer != (len(self.hparams['ae_decoding_n_channels'])-1) \
                        and self.hparams['ae_batch_norm']:
                    self.decoder.append(keras.layers.BatchNormalization(
                        name='bn' + str(global_layer_num),
                        axis=3,
                        momentum=self.hparams['ae_batch_norm_momentum']))

                # nonlinearity
                if i_layer == (len(self.hparams['ae_decoding_n_channels'])-1) \
                        and not self.hparams['ae_decoding_last_FF_layer']:
                    self.decoder.append(keras.layers.Activation(
                        tf.nn.sigmoid,
                        name='sigmoid' + str(global_layer_num)))
                else:
                    self.decoder.append(keras.layers.Activation(
                        tf.nn.leaky_relu,
                        name='relu' + str(global_layer_num)))

                global_layer_num += 1

        # Optional final FF layer (rarely used)
        if self.hparams['ae_decoding_last_FF_layer']:  # have last layer be feedforward if this is 1
            output_size = self.hparams['ae_decoding_x_dim'][-1] * \
                          self.hparams['ae_decoding_y_dim'][-1] * \
                          self.hparams['ae_decoding_n_channels'][-1] * \
                          self.hparams['ae_input_dim'][0] * \
                          self.hparams['ae_input_dim'][1] * \
                          self.hparams['ae_input_dim'][2]

            self.decoder.append(keras.layers.Dense(
                output_size,
                name='last_FF' + str(global_layer_num) + '',
                activation=tf.nn.sigmoid))

        if self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')

    def forward(self, x):

        # First FF layer/resize to be convolutional input
        img_shape = (
            -1,
            self.hparams['ae_decoding_starting_dim'][1],
            self.hparams['ae_decoding_starting_dim'][2],
            self.hparams['ae_decoding_starting_dim'][0])
        x = self.ff(x)
        x = tf.reshape(x, img_shape)

        for layer in self.decoder:
            if isinstance(layer, keras.layers.Conv2DTranspose):
                x = layer.apply(x)
            elif isinstance(layer, keras.layers.Dense):
                x = tf.contrib.layers.flatten(x)
                x = layer.apply(x)
                img_shape = (
                    -1,
                    self.hparams['ae_input_dim'][1],
                    self.hparams['ae_input_dim'][2],
                    self.hparams['ae_input_dim'][0])
                x = tf.reshape(x, img_shape)
            else:
                x = layer.apply(x)
            print(layer.name)

        if self.hparams['model_class'] == 'ae':
            self.y = x
        elif self.hparams['model_class'] == 'vae':
            raise ValueError('Not Implemented Error')
        else:
            raise ValueError('Not Implemented Error')

        return self.y


class LinearAEEncoder(object):

    def __init__(self, n_latents, input_size):
        """

        Args:
            n_latents (int):
            input_size (list or tuple): n_channels x y_pix x x_pix
        """
        super().__init__()

        self.n_latents = n_latents
        self.input_size = input_size
        self.build_model()

    def build_model(self):
        self.encoder = [keras.layers.Dense(
            units=self.n_latents,
            input_shape=(None, np.prod(self.input_size)))]

    def forward(self, x):
        # reshape
        x = tf.contrib.layers.flatten(x)
        for layer in self.encoder:
            x = layer.apply(x)
        return x


class LinearAEDecoder(object):

    def __init__(self, n_latents, output_size):
        """

        Args:
            n_latents (int):
            output_size (list or tuple): n_channels x y_pix x x_pix
            encoder (nn.Module object): for linking encoder/decoder weights
        """
        super().__init__()
        self.n_latents = n_latents
        self.output_size = output_size
        self.build_model()

    def build_model(self):

        self.decoder = [keras.layers.Dense(
            units=np.prod(self.output_size),
            input_shape=(None, self.n_latents))]

    def forward(self, x):
        for layer in self.decoder:
            x = layer.apply(x)
        x = tf.reshape(x, (-1, *self.output_size))
        return x


class DenseTranspose(keras.layers.Dense):
    """
    A Keras dense layer that has its weights set to be the transpose of
    another layer. Used for implemeneting BidNNs.
    """
    def __init__(self, other_layer, **kwargs):
        super().__init__(other_layer.input_shape, **kwargs)
        self.other_layer = other_layer

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        # self.input_spec = [InputSpec(dtype=K.floatx(),
        #                              ndim='2+')]

        self.W = tf.transpose(self.other_layer.W)

        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True


class LinearAEDecoderTie(object):

    def __init__(self, n_latents, output_size, encoder):
        """

        Args:
            n_latents (int):
            output_size (list or tuple): n_channels x y_pix x x_pix
            encoder (nn.Module object): for linking encoder/decoder weights
        """
        super().__init__()
        self.n_latents = n_latents
        self.output_size = output_size
        self.encoder = encoder
        self.build_model()

    def build_model(self):
        self.decoder = [DenseTranspose(self.encoder.encoder[0])]

    def forward(self, x):
        for layer in self.decoder:
            x = layer.apply(x)
        x = tf.reshape(x, (-1, *self.output_size))
        return x


class AE(object):

    def __init__(self, hparams):

        tf.reset_default_graph()

        super().__init__()
        self.hparams = hparams
        self.model_type = self.hparams['model_type']
        self.img_size = (
                self.hparams['y_pixels'],
                self.hparams['x_pixels'],
                self.hparams['n_input_channels'])

        # get model input/output
        self.encoder_input = tf.placeholder(
            dtype=tf.float32, shape=(None, *self.img_size))

        # add in mask (for cabled data)
        if hparams['use_output_mask']:
            self.mask = tf.placeholder(
                dtype=tf.float32, shape=(None, *self.img_size))
        else:
            self.mask = None

        self.build_model()
        self.forward()

    def build_model(self):

        if self.model_type == 'conv':
            self.encoding = ConvAEEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            n_latents = self.hparams['n_ae_latents']
            self.encoding = LinearAEEncoder(n_latents, self.img_size)
            self.decoding = LinearAEDecoder(n_latents, self.img_size)
        elif self.model_type == 'linear-tie':
            n_latents = self.hparams['n_ae_latents']
            self.encoding = LinearAEEncoder(n_latents, self.img_size)
            self.decoding = LinearAEDecoderTie(n_latents, self.img_size, self.encoding)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

    def forward(self):

        self.x = self.encoding.forward(self.encoder_input)
        self.decoder_input = tf.placeholder_with_default(
            self.x, shape=(None, self.hparams['n_ae_latents']))
        self.y = self.decoding.forward(self.decoder_input)

        return self.y, self.x

    def to(self, device):
        pass
