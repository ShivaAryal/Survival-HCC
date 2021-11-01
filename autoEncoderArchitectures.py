import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras import metrics, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Lambda, Activation


class LadderCallback(Callback):
    # for ladder autoencoder
    # A callback on each epoch end, increments beta by kappa

    def __init__(self, beta, kappa, max_val=1):
        self.beta = beta
        self.kappa = kappa
        self.max_val = max_val

    def on_epoch_end(self, *args, **kwargs):
        if K.get_value(self.beta) <= self.max_val:
            K.set_value(self.beta, K.get_value(self.beta)+self.kappa)


def train_model(vae, x_train, epochs, batch_size, x_val, beta, kappa, max_beta_val, verbose=0):
    hist = vae.fit(
        np.array(x_train),
        shuffle=True,
        epochs=epochs,
        verbose=verbose,
        batch_size=batch_size,
        validation_data=(np.array(x_val), None),
        callbacks=[LadderCallback(beta, kappa, max_beta_val)]
    )
    return hist


def deep_vae(
    input_dim,
    hidden_dims=None,
    latent_dim=100,
    initial_beta_val=0,
    learning_rate=0.0005,
    epsilon_std=1.0,
    kappa=1.0,
    epochs=50,
    batch_size=50,
    batch_normalize_inputs=True,
    batch_normalize_embedding=False,
    relu_embedding=False,
    max_beta_val=1
):
    if hidden_dims is None:
        hidden_dims = [300]

        # function for making model differentiable

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=tf.shape(
            z_mean), mean=0.0, stddev=epsilon_std)

        z = z_mean + K.exp(z_log_var / 2) * epsilon

        return z

    beta = K.variable(initial_beta_val)

    original_dim = input_dim

    rnaseq_input = Input(shape=(original_dim,), name="input")

    if batch_normalize_inputs:
        batchnorm_input = BatchNormalization(
            name="batchnorm_input")(rnaseq_input)
    else:
        batchnorm_input = rnaseq_input

    prev = batchnorm_input
    if hidden_dims:
        for i, hidden_dim in enumerate(hidden_dims):
            z = Dense(hidden_dim, activation="relu", name=f"hidden_{i}")(prev)
            prev = z
    else:
        z = prev

    l_mean_component = Dense(
        latent_dim, kernel_initializer="glorot_uniform", name="latent_mean"
    )
    l_mean_dense_linear = l_mean_component(z)

    if batch_normalize_embedding:
        l_mean_dense_batchnorm = BatchNormalization(
            name="batchnorm_latent_mean")(l_mean_dense_linear)
    else:
        l_mean_dense_batchnorm = l_mean_dense_linear

    if relu_embedding:
        l_mean_encoded = Activation("relu", name="relu_latent_mean")(
            l_mean_dense_batchnorm
        )
    else:
        l_mean_encoded = l_mean_dense_batchnorm

    l_log_var_dense_linear = Dense(
        latent_dim, kernel_initializer="glorot_uniform", name="latent_log_var"
    )(z)

    if batch_normalize_embedding:
        l_log_var_dense_batchnorm = BatchNormalization(name="batchnorm_latent_log_var")(
            l_log_var_dense_linear
        )
    else:
        l_log_var_dense_batchnorm = l_log_var_dense_linear

    if relu_embedding:
        l_log_var_encoded = Activation("relu", name="relu_latent_log_var")(
            l_log_var_dense_batchnorm
        )
    else:
        l_log_var_encoded = l_log_var_dense_batchnorm

    l = Lambda(sampling, output_shape=(latent_dim,), name="sample_latent")(
        [l_mean_encoded, l_log_var_encoded]
    )

    encoder_target = l_mean_component(z)
    if batch_normalize_embedding:
        encoder_target = BatchNormalization(name="batchnorm_encoder_target")(
            encoder_target
        )
    if relu_embedding:
        encoder_target = Activation(
            "relu", name="relu_encoder_target")(encoder_target)

    # decoder latent->hidden
    if hidden_dims:
        prev = l
        for i, hidden_dim in reversed(list(enumerate(hidden_dims))):
            h = Dense(
                hidden_dim,
                kernel_initializer="glorot_uniform",
                activation="relu",
                name=f"decode_hidden_{i}",
            )(prev)
            prev = h
    else:
        h = l

    reconstruction = Dense(
        original_dim,
        kernel_initializer="glorot_uniform",
        activation="sigmoid",
        name="reconstruction",
    )(h)

    adam = optimizers.Adam(learning_rate=learning_rate)
    vae = Model(rnaseq_input, reconstruction)

    reconstruction_loss = original_dim * metrics.binary_crossentropy(
        rnaseq_input, reconstruction
    )
    kl_loss = -0.5 * K.sum(
        1 + l_log_var_encoded -
        K.square(l_mean_encoded) - K.exp(l_log_var_encoded),
        axis=-1,
    )
    vae_loss = K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))
    vae.add_loss(vae_loss)
    vae.compile(optimizer=adam)

    # non-sampling encoder
    encoder = Model(rnaseq_input, encoder_target)

    # sampling encoder
    sampling_encoder = Model(rnaseq_input, l)

    # create a decoder model
    encoded_input = Input(shape=(latent_dim,))
    prev = encoded_input
    if hidden_dims:
        for i in reversed(range(len(hidden_dims) + 1)):
            prev = vae.layers[-(i + 1)](prev)
    decoder = Model(encoded_input, prev)

    return vae, encoder, sampling_encoder, decoder, beta
