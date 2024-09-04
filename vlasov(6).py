import time
import tensorflow as tf
import numpy as np
import math
import scipy.optimize
from pyDOE import lhs


class NN(tf.keras.models.Model):
    def __init__(self, layers):
        super().__init__()
        self.layers_config = layers

    def Initialize(self, layers):
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(tf.keras.layers.Dense(layers[i+1], activation=tf.nn.tanh, kernel_initializer='glorot_uniform'))
        return layer_list

    def forward(self, X, layer_list, lb, ub):
        H = 2.0 * (X - lb) / (ub - lb) - 1.0  # Normalization of input
        for i in range(len(layer_list) - 1):
            H = layer_list[i](H)
        H = layer_list[-1](H)
        return H


class vlasov1DNN:

    def __init__(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0,
                 numSteps, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        self.lb = lb
        self.ub = ub
        self.lbE = lb[0:2]
        self.ubE = ub[0:2]
        self.layers = layers
        self.layersE = layersE
        self.numSteps = numSteps
        self.N_inner = N_inner
        self.N_i = N_i
        self.N_b = N_b
        self.N_eta = N_eta
        self.N_E = N_E
        self.N_E0 = N_E0
        self.N_Eb = N_Eb

        self.iters = 0

        # Inputs
        self.X_inner = X_inner
        self.E_inner = E_inner
        self.X_i = X_i
        self.f_i = f_i
        self.X_b = X_b
        self.X_t = X_t
        self.X_eta = X_eta
        self.E_b = E_b
        self.E_t = E_t
        self.X_E0 = X_E0
        self.E_0 = E_0

        self.nn1 = NN(layers)
        self.nn2 = NN(layersE)
        self.nn_layers = self.nn1.Initialize(self.layers)
        self.nn_layers_e = self.nn2.Initialize(self.layersE)

    def net_eta(self, t, x, v):
        eta = self.nn1.forward(tf.stack([t, x, v], axis=1), self.nn_layers, self.lb, self.ub)
        return tf.squeeze(eta)

    def net_phi(self, t, x):
        phi = self.nn2.forward(tf.stack([t, x], axis=1), self.nn_layers_e, self.lbE, self.ubE)
        return tf.squeeze(phi)

    def net_E(self, t, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            phi = self.net_phi(t, x)
        E = -tape.gradient(phi, x)
        return E

    def net_E_d(self, t, x):
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                phi = self.net_phi(t, x)
            E = tape.gradient(phi, x)
        E_x = tape2.gradient(E, x)
        return E_x

    def net_f(self, t, x, v):
        with tf.GradientTape() as tape:
            tape.watch(v)
            eta = self.net_eta(t, x, v)
        f = tape.gradient(eta, v)
        return f

    def net_f_d(self, t, x, v):
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape:
                tape.watch(v)
                eta = self.net_eta(t, x, v)
            f = tape.gradient(eta, v)
        f_x = tape2.gradient(f, x)
        return f_x

    def net_N(self, t, x, v):
        with tf.GradientTape() as tape3:
            tape3.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(t)
                with tf.GradientTape() as tape:
                    tape.watch(v)
                    eta = self.net_eta(t, x, v)
                f_v = tape.gradient(eta, v)
            f_x = tape2.gradient(f_v, x)
        f_t = tape3.gradient(f_x, t)
        phi = self.net_phi(t, x)
        E = -tape3.gradient(phi, x)
        N = f_t + v * f_x - E * f_v
        return N

    def net_gauss(self, t, x, v):
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                phi = self.net_phi(t, x)
            E = -tape.gradient(phi, x)
        E_x = tape2.gradient(E, x)
        n_e = self.net_eta(t, x, v) - self.net_eta(t, x, -3.5 * tf.ones(tf.shape(v)[0]))
        n_i = 1
        return 4 * math.pi * (n_i - n_e) - E_x

    @staticmethod
    def generator(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0, N_inner, N_i, N_b, N_eta, N_E, N_Eb,
                  N_E0):
        idx_inner = np.random.choice(X_inner.shape[0], N_inner, replace=False)
        idx_i = np.random.choice(X_i.shape[0], N_i, replace=False)
        idx_b = np.random.choice(X_b.shape[0], N_b, replace=False)
        idx_eta = np.random.choice(X_eta.shape[0], N_eta, replace=False)
        idx_E = np.random.choice(E_inner.shape[0], N_E, replace=False)
        idx_Eb = np.random.choice(E_b.shape[0], N_Eb, replace=False)
        idx_E0 = np.random.choice(X_E0.shape[0], N_E0, replace=False)

        yield (
            X_inner[idx_inner, :],
            X_i[idx_i, :],
            f_i[idx_i],
            X_b[idx_b, :],
            X_t[idx_b, :],
            X_eta[idx_eta, :],
            E_inner[idx_E, :],
            E_b[idx_Eb, :],
            E_t[idx_Eb, :],
            X_E0[idx_E0, :],
            E_0[idx_E0]
        )

    def func(self, X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0):

        self.N_pred = self.net_N(X_inner[:, 0], X_inner[:, 1], X_inner[:, 2])
        self.f_i_pred = self.net_f(X_i[:, 0], X_i[:, 1], X_i[:, 2])
        self.f_b_pred = self.net_f(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.f_t_pred = self.net_f(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.f_db_pred = self.net_f_d(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.f_dt_pred = self.net_f_d(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.f_d_pred = self.net_f_d(X_i[:, 0], X_i[:, 1], X_i[:, 2])
        self.eta_pred = self.net_eta(X_eta[:, 0], X_eta[:, 1], X_eta[:, 2])
        self.eta_t_pred = self.net_eta(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.eta_b_pred = self.net_eta(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.eta_i_pred = self.net_eta(X_i[:, 0], X_i[:, 1], X_i[:, 2])
        self.E_pred = self.net_E(E_inner[:, 0], E_inner[:, 1])
        self.gauss_pred = self.net_gauss(E_inner[:, 0], E_inner[:, 1], E_inner[:, 2])
        self.E_b_pred = self.net_E(E_b[:, 0], E_b[:, 1])
        self.E_t_pred = self.net_E(E_t[:, 0], E_t[:, 1])
        self.E_0_pred = self.net_E(X_E0[:, 0], X_E0[:, 1])

        loss = tf.reduce_mean(tf.square(f_i - self.f_i_pred)) \
               + tf.reduce_mean(tf.square(self.f_db_pred)) \
               + tf.reduce_mean(tf.square(self.f_dt_pred)) \
               + tf.reduce_mean(tf.square(self.f_d_pred)) \
               + tf.reduce_mean(tf.square(self.eta_t_pred)) \
               + tf.reduce_mean(tf.square(self.eta_b_pred)) \
               + tf.reduce_mean(tf.square(self.eta_i_pred)) \
               + tf.reduce_mean(tf.square(self.E_b_pred)) \
               + tf.reduce_mean(tf.square(self.E_t_pred)) \
               + tf.reduce_mean(tf.square(self.E_0_pred)) \
               + tf.reduce_mean(tf.square(self.gauss_pred)) \
               + tf.reduce_mean(tf.square(self.N_pred))
        return loss

    def train_with_lbfgs(self):
        trainable_vars = self.nn_layers + self.nn_layers_e

        optimizer = tf.keras.optimizers.Adam()
        self.func = tf.function(self.func)

        @tf.function
        def grad_fn(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0):
            with tf.GradientTape() as tape:
                loss = self.func(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
            grads = tape.gradient(loss, trainable_vars)
            return loss, grads

        for step, (X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0) in enumerate(
                self.generator(self.X_inner, self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta, self.E_inner,
                               self.E_b, self.E_t, self.X_E0, self.E_0, self.N_inner, self.N_i, self.N_b,
                               self.N_eta, self.N_E, self.N_Eb, self.N_E0)):
            if step >= self.numSteps:
                break
            loss, grads = grad_fn(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            if step % 100 == 0:
                print(f"Step: {step}, Loss: {loss.numpy()}")

        return loss
