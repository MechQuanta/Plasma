import time

import tensorflow as tf
import numpy as np
import math
import scipy.optimize
from pyDOE import lhs


class NN(tf.keras.models.Model):
    def __init__(self,layers):
        super().__init__()
        self.layers = layers
        self.layersE = layersE
    def Initialize(self,layers):
        layer_list = []
        for i in range(len(layers) - 1):
            tf.keras.layers.Dense(layers[i],layers[i+1], activation=tf.nn.tanh,kernel_initializer='glorot_uniform')
        return layer_list

    def forward(self, X, layer_list, lb, ub):
        num_layers = len(layer_list) + 1
        H = 2.0 * (X - lb) / (ub - lb) + 1.0
        for i in range(0,num_layers-2):
            H = layer_list[i](H)
        H = layer_list[-1](H)
        return H

class vlasov1DNN:

    def __init__(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0,numSteps,N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        self.lb = lb
        self.ub = ub
        self.lbE = lb[0:2]
        self.ubE = ub[0:2]
        self.layers = layers
        self.layersE = layersE
        self.numSteps=numSteps
        self.N_inner=N_inner
        self.N_i=N_i
        self.N_b=N_b
        self.N_eta=N_eta
        self.N_E=N_E
        self.N_E0=N_E0
        self.N_Eb=N_Eb

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
        self.nn2= NN(layersE)
        self.nn1.build()
        self.nn_layers = self.nn1.Initialize(self.layers)
        self.nn_layers_e = self.nn2.Initialize(self.layersE)


    def net_eta(self,t,x,v):
        eta = self.nn.forward(tf.stack([t,x,v],axis=1),self.nn_layers,self.lb,self.ub)
        return tf.squeeze(eta)
    def net_phi(self,t,x):
        phi = self.nn.forward(tf.stack([t,x],axis=1),self.nn_layers_e,self.lbE,self.ubE)
        return tf.squeeze(phi)
    def net_E(self,t,x):
        phi = self.net_phi(t,x)
        E = -tf.gradients(phi,x)[0]
        return E
    def net_E_d(self,t,x):
        phi = self.net_phi(t,x)
        E = tf.gradients(phi,x)[0]
        E_x = tf.gradients(E,x)[0]
        return E_x
    def net_f(self,t,x,v):
        eta = self.net_eta(t,x,v)
        f = tf.gradients(eta,v)[0]
        return f
    def net_f_d(self,t,x,v):
        eta = self.net_eta(t,x,v)
        f = tf.gradients(eta,v)[0]
        f_x = tf.gradients(f,x)[0]
        return f_x
    def net_N(self,t,x,v):
        eta = self.net_eta(t,x,v)
        f = tf.gradients(eta,v)[0]
        f_t = tf.gradients(f,t)[0]
        f_x = tf.gradients(f,x)[0]
        f_v = tf.gradients(f,v)[0]
        phi = self.net_phi(t,x)
        E = -tf.gradients(phi,x)[0]
        N = f_t + v*f_x - E*f_v
        return N
    def net_gauss(self,t,x,v):
        phi = self.net_phi(t,x)
        E = -tf.gradients(phi,x)[0]
        E_x = tf.gradients(E,x)[0]
        n_e = self.net_eta(t,x,v) - self.net_eta(t,x,-3.5*tf.ones(tf.shape(v)[0]))
        n_i = 1
        return 4*math.pi*(n_i-n_e) - E_x

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

    def func(self,X_inner,X_i,f_i,X_b,X_t,X_eta,E_inner,E_b,E_t,X_E0,E_0):


        self.N_pred = self.net_N(X_inner[:, 0], X_inner[:, 1], X_inner[:, 2])
        self.f_i_pred = self.net_f(X_i[:, 0], X_i[:, 1], X_i[:, 2])
        self.f_b_pred = self.net_f(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.f_t_pred = self.net_f(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.f_db_pred = self.net_f_d(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.f_dt_pred = self.net_f_d(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.f_d_pred = self.net_f_d(X_i[:, 0], X_i[:, 1], X_i[:.2])
        self.eta_pred = self.net_eta(X_eta[:, 0], X_eta[:, 1], X_eta[:, 2])
        self.eta_t_pred = self.net_eta(X_t[:, 0], X_t[:, 1], X_t[:, 2])
        self.eta_b_pred = self.net_eta(X_b[:, 0], X_b[:, 1], X_b[:, 2])
        self.gauss_pred = self.net_gauss(E_inner[:, 0], E_inner[:, 1], E_inner[:, 2])
        self.phi_b_pred = self.net_phi(E_b[:, 0], E_b[:, 1])
        self.phi_t_pred = self.net_phi(E_t[:, 0], E_t[:, 1])
        self.E_b_pred = self.net_E(E_b[:, 0], E_b[:, 1])
        self.E_t_pred = self.net_E(E_t[:, 0], E_t[:, 1])
        self.E_db_pred = self.net_E_d(E_b[:, 0], E_b[:, 1])
        self.E_dt_pred = self.net_E_d(E_t[:, 0], E_t[:, 1])
        self.E_i_pred = self.net_E(X_E0[:, 0], X_E0[:, 1])
        return self.N_pred, self.f_i_pred, self.f_b_pred, self.f_t_pred, self.f_db_pred, self.f_dt_pred,self.eta_pred,self.eta_t_pred,self.eta_b_pred,self.gauss_pred,self.phi_b_pred,self.phi_t_pred,self.E_b_pred,self.E_t_pred,self.E_db_pred,self.E_dt_pred,self.E_i_pred,self.E_0


    def loss_fn(self,numSteps ,N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        for i in range(numSteps):
            for (X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0) in self.generator(self.X_inner, self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta, self.E_inner, self.E_b, self.E_t, self.X_E0, self.E_0,N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
                self.N_pred, self.f_i_pred, self.f_b_pred, self.f_t_pred, self.f_db_pred, self.f_dt_pred,self.eta_pred,self.eta_t_pred,self.eta_b_pred,self.gauss_pred,self.phi_b_pred,self.phi_t_pred,self.E_b_pred,self.E_t_pred,self.E_db_pred,self.E_dt_pred,self.E_i_pred,self.E_0 = self.func(self.X_inner,self.X_i,self.f_i,self.X_b,self.X_t,self.X_eta,self.E_inner,self.E_b,self.E_t,self.X_E0,self.E_0)

            self.NLoss = tf.reduce_mean(tf.square(self.N_pred))
            self.gaussLoss = tf.reduce_mean(tf.square(self.gauss_pred))
            self.iLoss = tf.reduce_mean(tf.square(self.f_i_pred - self.f_i))
            self.bLoss = tf.reduce_mean(tf.square(self.f_t_pred - self.f_b_pred)) + \
                         tf.reduce_mean(tf.square(self.f_dt_pred - self.f_db_pred))
            # .5*tf.reduce_mean(tf.square(self.eta_t_pred - self.eta_b_pred))
            # self.etaLoss = tf.reduce_mean(tf.square(self.eta_pred))
            self.EbLoss = tf.reduce_mean(tf.square(self.E_b_pred - self.E_t_pred)) + \
                          tf.reduce_mean(tf.square(self.E_db_pred - self.E_dt_pred)) + \
                          tf.reduce_mean(tf.square(self.phi_b_pred - self.phi_t_pred))
            self.E0loss = tf.reduce_mean(tf.square(self.E_0 - self.E_i_pred))
            self.loss = self.iLoss + self.bLoss + self.NLoss + \
                        self.gaussLoss + self.EbLoss + self.E0loss # + self.etaLoss
            return self.loss
    def compute_grad(self):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.numSteps,self.N_inner,self.N_i,self.N_b.N_eta,N_E,N_Eb,N_E0)
            grads = tape.gradient(loss, self.nn.trainable_variables())
        return loss,grads
    def callback(self, loss, NLoss, iLoss, bLoss, gaussLoss, EbLoss, E0Loss):
        self.iters += 1
        if self.iters % 50 == 0:
            print('Loss: {}'.format(loss))
            print('Interior loss: {} Initial loss: {} Boundary loss: {}'.format(NLoss, iLoss, bLoss))
            print('E0 loss: {} Gauss\'s loss: {} E Boundary loss: {}'.format(E0Loss, gaussLoss, EbLoss))

    def train_with_lbfgs(self,max_num):

        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in self.nn.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec

        w0 = vec_weight().numpy()

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            idx = 0
            for v in self.nn.trainable_variables:
                vs = v.shape

                # weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    updated_val = tf.reshape(weight_vec[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # bias vectors
                elif len(vs) == 1:
                    updated_val = weight_vec[idx:idx + vs[0]]
                    idx += vs[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))

        def loss_grad(w):
            # update weights in model
            restore_weight(w)
            loss, grads = self.compute_grad()
            # vectorize gradients
            grad_vec = []
            for g in grads:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = loss.numpy().astype(np.float64)
            self.instant_loss = loss
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec

        return scipy.optimize.minimize(fun=loss_grad,
                                       x0=w0,
                                       jac=True,
                                       method='L-BFGS-B',
                                       # method='BFGS',
                                       callback=self.callback,
                                       options={'maxiter': max_num,
                                                'maxfun': 10000,
                                                'maxcor': 200,
                                                'maxls': 200,
                                                'gtol': np.nan,  # 1.0 * np.finfo(float).eps,#np.nan,
                                                'ftol': np.nan})  # 1.0 * np.finfo(float).eps})#np.nan})

    @staticmethod
    def initialConditions(x, v, alpha, V_T):
        return np.exp(-0.5 * (x ** 2 + v ** 2) / (alpha * V_T) ** 2)

    @staticmethod
    def Einitial(x, alpha):
        return np.sin(math.pi * x) * alpha


if __name__ == '__main__':


    v_max = 3.5
    V_T = 1
    alpha = 0.2
    t_max = 1.
    pi = math.pi
    x_max = pi
    N_pred = 150
    N_init = 150
    N_E = 10000
    N_N = 100000
    N_Eb = 2000
    N_Einit = 2000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 1]
    layersE = [2, 15, 15, 15, 15, 15, 1]

    x = np.linspace(-x_max, x_max, N_pred)
    t = np.linspace(0, t_max, N_pred)
    v = np.linspace(-v_max, v_max, N_pred)

    T, X, V = np.meshgrid(t, x, v)
    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None], V.flatten()[:, None]))
    lb = X_star.min(0)
    ub = X_star.max(0)

    print("lower_bound: ", lb)
    print("upper_bound: ", ub)

    xx2 = np.stack((T[0, :, :], X[0, :, :], V[0, :, :]), axis=2)  # x=-pi condition
    xx2 = np.reshape(xx2, (N_pred ** 2, 3))
    print("at -x_max position: ", xx2.shape)
    xx3 = np.stack((T[-1, :, :], X[-1, :, :], V[-1, :, :]), axis=2)
    xx3 = np.reshape(xx3, (N_pred ** 2, 3))
    print("at x_max position: ", xx3.shape)

    xx4 = np.stack((T[:, :, 0], X[:, :, 0], V[:, :, 0]), axis=2)
    xx4 = np.reshape(xx4, (N_pred ** 2, 3))
    print("at -v_max position: ", xx4.shape)
    # plt.plot(xx4[:,0],xx4[:,1],label='-x_max to x_max and time')
    # plt.xlabel('time')
    # plt.ylabel('-x_max to x_max position')
    # plt.legend()
    # plt.savefig('Vlasov_Equation.png')
    # plt.show()

    xx1 = np.stack((np.zeros(N_init ** 2), np.linspace(-x_max, x_max, N_init ** 2),
                    np.clip(np.squeeze(np.random.multivariate_normal([0.], [[2.]], N_init ** 2)), -v_max, v_max)),
                   axis=1)
    ff1 = vlasov1DNN.initialConditions(xx1[:, 1], xx1[:, 2], alpha, V_T)

    X_f_train = np.vstack([xx1, xx2, xx3])

    print("lowe_bound: ", lb)
    print("upper_bound: ", ub)
    print("upper - lower : ", ub - lb)
    print("lb + (ub - lb): ", lb + (ub - lb))
    print(X_f_train.shape)

    X_N_train = lb + (ub - lb) * lhs(3, N_N)
    X_N_train = np.vstack((X_N_train, X_f_train))

    print(X_N_train.shape)

    xbE = np.stack((np.linspace(0, t_max, N_Eb), -x_max * np.ones(N_Eb)), axis=1)
    xtE = np.stack((np.linspace(0, t_max, N_Eb), x_max * np.ones(N_Eb)), axis=1)
    print(xbE.shape)

    X_E0 = np.stack((np.zeros(N_Einit), np.linspace(-x_max, x_max, N_Einit)), axis=1)
    E_0 = vlasov1DNN.Einitial(X_E0[:, 1], alpha)
    print("Lower_bounds: ", lb[0:2])
    print("Upper_bounds: ", ub[0:2])
    X_E = lb[0:2] + (ub[0:2] - lb[0:2]) * lhs(2, N_E)
    X_E = np.vstack((X_E, xbE, xtE))
    print(X_E.shape)
    X_E = np.stack((X_E[:, 0], X_E[:, 1], v_max * np.ones(N_E + 2 * N_Eb)), axis=1)
    N_inner = 64000
    N_i = 16000
    N_b = 16000
    N_eta = 0
    N_E =4000
    N_Eb =1000
    N_E0 = 1000
    numSteps =3
    model = vlasov1DNN(lb, ub, layers, layersE, X_N_train, X_E, xx1, ff1, xx2, xx3, xx4, xbE, xtE, X_E0, E_0,numSteps,N_inner,N_i,N_b,N_eta,N_E,N_Eb,N_E0)

    with tf.device('/device:GPU:0'):
        start_time = time.time()
        model.train_with_lbfgs(200000)  # N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))


