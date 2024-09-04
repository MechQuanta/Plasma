import tensorflow as tf
import numpy as np
import math
import tensorflow_probability as tfp


class vlasov1DNN(tf.Module):

    def _init_(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0,
               generator):
        self.lb = lb
        self.ub = ub
        self.lbE = lb[0:2]
        self.ubE = ub[0:2]
        self.layers = layers
        self.layersE = layersE
        self.generator = generator
        self.iters = 0

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.Eweights, self.Ebiases = self.initialize_NN_E(self.layersE)

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



    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def initialize_NN_E(self, layersE):
        weights = []
        biases = []
        num_layers = len(layersE)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layersE[l], layersE[l + 1]])
            b = tf.Variable(tf.zeros([1, layersE[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(6 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def forward(self, X, weights, biases, lb, ub):
        num_layers = len(weights) + 1
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_eta(self, t, x, v):
        eta = self.forward(tf.stack([t, x, v], axis=1), self.weights, self.biases, self.lb, self.ub)
        return tf.squeeze(eta)

    def net_phi(self, t, x):
        phi = self.forward(tf.stack([t, x], axis=1), self.Eweights, self.Ebiases, self.lbE, self.ubE)
        return tf.squeeze(phi)

    def net_E(self, t, x):
        phi = self.net_phi(t, x)
        E = -tf.gradients(phi, x)[0]
        return E

    def net_E_d(self, t, x):
        E = self.net_E(t, x)
        E_x = tf.gradients(E, x)[0]
        return E_x

    def net_f(self, t, x, v):
        eta = self.net_eta(t, x, v)
        f = tf.gradients(eta, v)[0]
        return f

    def net_f_d(self, t, x, v):
        f = self.net_f(t, x, v)
        f_x = tf.gradients(f, x)[0]
        return f_x

    def net_N(self, t, x, v):
        f = self.net_f(t, x, v)
        f_t = tf.gradients(f, t)[0]
        f_x = tf.gradients(f, x)[0]
        f_v = tf.gradients(f, v)[0]
        phi = self.net_phi(t, x)
        E = -tf.gradients(phi, x)[0]
        N = f_t - E * f_v + v * f_x
        return N

    def net_gauss(self, t, x, v):
        phi = self.net_phi(t, x)
        E = -tf.gradients(phi, x)[0]
        E_x = tf.gradients(E, x)[0]
        n_e = self.net_eta(t, x, v) - self.net_eta(t, x, -3.5 * tf.ones(tf.shape(v)[0]))
        n_i = 1.0
        return 4 * math.pi * (n_i - n_e) - E_x

    def loss_fn(self, X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0):
        N_pred =tf.reduce_mean(tf.square(self.net_N(X_inner[:, 0], X_inner[:, 1], X_inner[:, 2])))
        gauss_pred = tf.reduce_mean(tf.square(self.net_gauss(E_inner[:, 0], E_inner[:, 1], E_inner[:, 2])))
        iLoss = tf.reduce_mean(tf.square(self.net_f(X_i[:, 0], X_i[:, 1], X_i[:, 2]) - f_i))
        bLoss = tf.reduce_mean(tf.square(self.net_f(X_t[:, 0], X_t[:, 1], X_t[:, 2]) - self.net_f(X_b[:, 0], X_b[:, 1], X_b[:, 2])))+tf.reduce_mean(tf.square(self.net_f_d(X_t[:,0],X_t[:,1],X_t[:,2])-self.net_f_d(X_b[:,0],X_b[:,1],X_b[:,2])))
        EbLoss = tf.reduce_mean(tf.square(self.net_E(E_b[:, 0], E_b[:, 1]) - self.net_E(E_t[:, 0], E_t[:, 1]))) + tf.reduce_mean(tf.square(self.net_E_d(E_b[:, 0], E_b[:, 1]) - self.net_E_d(E_t[:, 0], E_t[:, 1]))) + tf.reduce_mean(tf.square(self.net_phi(E_b[:,0], E_b[:, 1]) - self.net_phi(E_t[:,0], E_t[:, 1])))
        E0loss = tf.reduce_mean(tf.square(E_0 - self.net_E(X_E0[:, 0], X_E0[:, 1])))
        loss = iLoss + bLoss + EbLoss + E0loss + tf.reduce_mean(tf.square(N_pred)) + gauss_pred
        return loss
    def train_lbfgs(self, X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0, max_iter=500):
        variables = self.trainable_variables

        def loss_and_grads():
            with tf.GradientTape() as tape:
                loss = self.loss_fn(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
            grads = tape.gradient(loss, variables)
            grads_and_vars = zip(grads, variables)
            return loss, grads_and_vars

        def func(x):
            # Update trainable variables with the new values from x
            tf.nest.map_structure(lambda var, val: var.assign(val), variables, tf.split(x, [tf.reduce_prod(var.shape) for var in variables]))
            loss, grads_and_vars = loss_and_grads()
            return loss.numpy().astype(np.float64), tf.concat([tf.reshape(grad, [-1]) for grad, _ in grads_and_vars], axis=0).numpy().astype(np.float64)

        x0 = tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0).numpy()

        result = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=x0,
            tolerance=1e-6,
            max_iterations=max_iter
        )

        # Update trainable variables with the optimized values
        tf.nest.map_structure(lambda var, val: var.assign(val), variables, tf.split(result.position, [tf.reduce_prod(var.shape) for var in variables]))

        print(f"L-BFGS-B optimization completed with status: {result.converged}")
        return result

    """
    @tf.function
    def train_step(self, X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
        """
    def train(self, numSteps, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        for step in range(numSteps):
            for X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0 in self.generator(
                    self.X_inner, self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta, self.E_inner,
                    self.E_b, self.E_t, self.X_E0, self.E_0, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):

                loss = self.train_lbfgs(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
                if step % 50 == 0:
                    print(f"Step: {step}, Loss: {loss.numpy()}")
    def generator(X_inner,X_i,f_i,X_b,X_t,X_eta,E_inner,E_b,E_t,X_E0,E_0,N_inner,N_i,N_b,N_eta,N_E,N_Eb,N_E0):
        v_max = 3.5
        def sampleRows(X,N):
            idx = np.random.choice(np.shape(X)[0],N,replace=False)
            return X[idx,:]
        X_inner = sampleRows(X_inner,N_inner)
        I = sampleRows(np.stack([X_i[:,0],X_i[:,1],X_i[:,2],f_i],axis=1),N_i)
        X_i= I[:,0:3]
        f_i = I[:,3]
        X = sampleRows(np.stack([X_b[:,0],X_b[:,1],X_b[:,2],X_t[:,0],X_t[:,1],X_t[:,2]],axis=1),N_b)
        X_b = X[:,0:3]
        X_t = X[:,3:]
        X_inner = np.vstack([X_inner, X_i, X_b, X_t])
        X_eta = sampleRows(X_eta,N_eta)
        E_inner = sampleRows(E_inner,N_inner)
        E = sampleRows(np.stack([E_b[:,0],E_b[:,1],E_t[:,0],E_t[:,1]],axis=1),N_Eb)
        E_b = E[:,0:2]
        E_t = E[:,2:]
        E_bv = np.stack((E_b[:,0],E_b[:,1],v_max*np.ones(N_Eb)),axis=1)
        E_tv = np.stack((E_t[:,0],E_t[:,1],v_max*np.ones(N_Eb)),axis=1)
        E_inner = np.vstack([E_inner,E_bv,E_tv])
        I = sampleRows(np.stack([X_E0[:,0],X_E0[:,1],E_0],axis=1),N_E0)
        X_E0 = I[:,0:2]
        E_0 = I[:,2]
        yield (X_inner,X_i,f_i,X_b,X_t,X_eta,E_inner,E_b,E_t,X_E0,E_0)
