import time

import tensorflow as tf
import numpy as np
import math
from scipy.optimize import minimize
from pyDOE import lhs


class Vlasov1DNN:

    def __init__(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0,generator):
        self.lb = lb
        self.ub = ub
        self.lbE = lb[0:2]
        self.ubE = ub[0:2]
        self.generator = generator
        self.layers = layers
        self.layersE = layersE
        self.iters = 0

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.Eweights, self.Ebiases = self.initialize_NN_E(self.layersE)

        # Convert inputs to tf.data.Dataset format for batching
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

        self.build_model()

    def build_model(self):
        self.optimizer = tf.keras.optimizers.Adam()

        # Define loss functions
        self.NLoss = tf.keras.losses.MeanSquaredError()
        self.gaussLoss = tf.keras.losses.MeanSquaredError()
        self.iLoss = tf.keras.losses.MeanSquaredError()
        self.bLoss = tf.keras.losses.MeanSquaredError()
        self.EbLoss = tf.keras.losses.MeanSquaredError()
        self.E0loss = tf.keras.losses.MeanSquaredError()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def initialize_NN_E(self, layersE):
        weights = []
        biases = []
        for l in range(len(layersE) - 1):
            W = self.xavier_init(size=[layersE[l], layersE[l+1]])
            b = tf.Variable(tf.zeros([1, layersE[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def kaiming_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        kaiming_stddev = np.sqrt(2 / (in_dim))
        return tf.Variable(tf.random.normal([in_dim, out_dim], stddev=kaiming_stddev), dtype=tf.float32)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(6 / (in_dim + out_dim))
        return tf.Variable(tf.random.normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def forward(self, X, weights, biases, lb, ub):
        num_layers = len(weights) + 1
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        for l in range(num_layers - 2):
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
        with tf.GradientTape() as tape:
            tape.watch(x)
            E = -tape.gradient(phi, x)
        return E

    def net_E_d(self, t, x):
        phi = self.net_phi(t, x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            E = -tape.gradient(phi, x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            E_x = tape.gradient(E, x)
        return E_x

    def net_f(self, t, x, v):
        eta = self.net_eta(t, x, v)
        with tf.GradientTape() as tape:
            tape.watch(v)
            f = tape.gradient(eta, v)
        return f

    def net_f_d(self, t, x, v):
        f = self.net_f(t, x, v)
        with tf.GradientTape() as tape:
            tape.watch(x)
            f_x = tape.gradient(f, x)
        return f_x

    def net_N(self, t, x, v):
        f = self.net_f(t, x, v)
        with tf.GradientTape() as tape:
            tape.watch(t)
            f_t = tape.gradient(f, t)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            f_x = tape2.gradient(f, x)
        with tf.GradientTape() as tape3:
            tape3.watch(v)
            f_v = tape3.gradient(f, v)
        phi = self.net_phi(t, x)
        E = -self.net_E(t, x)
        N = f_t - E * f_v + v * f_x
        return N

    def net_gauss(self, t, x, v):
        phi = self.net_phi(t, x)
        E = -self.net_E(t, x)
        E_x = self.net_E_d(t, x)
        n_e = self.net_eta(t, x, v) - self.net_eta(t, x, -3.5 * tf.ones(tf.shape(v)[0]))
        n_i = 1.0
        return 4 * math.pi * (n_i - n_e) - E_x

    def callback(self, loss, NLoss, iLoss, bLoss, gaussLoss, EbLoss, E0Loss):
        self.iters += 1
        if self.iters % 50 == 0:
            print('Loss: {}'.format(loss))
            print('Interior loss: {} Initial loss: {} Boundary loss: {}'.format(NLoss, iLoss, bLoss))
            print('E0 loss: {} Gauss\'s loss: {} E Boundary loss: {}'.format(E0Loss, gaussLoss, EbLoss))

    def train(self, numSteps, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        for i in range(numSteps):
            for (X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0) in self.generator(
                self.X_inner, self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta,
                self.E_inner, self.E_b, self.E_t, self.X_E0, self.E_0,
                N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):

                with tf.GradientTape() as tape:
                    # Calculate losses
                    N_pred = self.net_N(X_inner[:,0], X_inner[:,1], X_inner[:,2])
                    f_i_pred = self.net_f(X_i[:,0], X_i[:,1], X_i[:,2])
                    f_b_pred = self.net_f(X_b[:,0], X_b[:,1], X_b[:,2])
                    f_t_pred = self.net_f(X_t[:,0], X_t[:,1], X_t[:,2])
                    f_db_pred = self.net_f_d(X_b[:,0], X_b[:,1], X_b[:,2])
                    f_dt_pred = self.net_f_d(X_t[:,0], X_t[:,1], X_t[:,2])
                    f_d_pred = self.net_f_d(X_i[:,0], X_i[:,1], X_i[:,2])
                    eta_pred = self.net_eta(X_eta[:,0], X_eta[:,1], X_eta[:,2])
                    eta_t_pred = self.net_eta(X_eta[:,0], X_eta[:,1], X_eta[:,2])
                    phi_pred = self.net_phi(X_eta[:,0], X_eta[:,1])
                    E_pred = self.net_E(X_eta[:,0], X_eta[:,1])
                    E0_pred = self.net_phi(X_E0[:,0], X_E0[:,1])
                    gauss_pred = self.net_gauss(X_eta[:,0], X_eta[:,1], X_eta[:,2])
                    Eb_pred = self.net_E(X_eta[:,0], X_eta[:,1])
                    E0_pred = self.net_phi(X_E0[:,0], X_E0[:,1])

                    N_loss = self.NLoss(N_pred, tf.zeros_like(N_pred))
                    i_loss = self.iLoss(f_i_pred, f_i)
                    b_loss = self.iLoss(f_b_pred, tf.zeros_like(f_b_pred))
                    t_loss = self.iLoss(f_t_pred, tf.zeros_like(f_t_pred))
                    f_db_loss = self.iLoss(f_db_pred, tf.zeros_like(f_db_pred))
                    f_dt_loss = self.iLoss(f_dt_pred, tf.zeros_like(f_dt_pred))
                    f_d_loss = self.iLoss(f_d_pred, tf.zeros_like(f_d_pred))
                    eta_loss = self.iLoss(eta_pred, tf.zeros_like(eta_pred))
                    eta_t_loss = self.iLoss(eta_t_pred, tf.zeros_like(eta_t_pred))
                    phi_loss = self.iLoss(phi_pred, tf.zeros_like(phi_pred))
                    E_loss = self.iLoss(E_pred, tf.zeros_like(E_pred))
                    E0_loss = self.iLoss(E0_pred, tf.zeros_like(E0_pred))
                    gauss_loss = self.gaussLoss(gauss_pred, tf.zeros_like(gauss_pred))
                    Eb_loss = self.EbLoss(Eb_pred, tf.zeros_like(Eb_pred))
                    E0_loss = self.E0loss(E0_pred, tf.zeros_like(E0_pred))

                    loss = N_loss + i_loss + b_loss + t_loss + f_db_loss + f_dt_loss + f_d_loss + eta_loss + eta_t_loss + phi_loss + E_loss + E0_loss + gauss_loss + Eb_loss

                grads = tape.gradient(loss, self.weights + self.biases + self.Eweights + self.Ebiases)
                self.optimizer.apply_gradients(zip(grads, self.weights + self.biases + self.Eweights + self.Ebiases))

                self.callback(loss.numpy(), N_loss.numpy(), i_loss.numpy(), b_loss.numpy(), gauss_loss.numpy(), Eb_loss.numpy(), E0_loss.numpy())


    @staticmethod
    def initialConditions(x, v, alpha, V_T):
        return np.exp(-0.5 * (x ** 2 + v ** 2) / (alpha * V_T) ** 2)

    @staticmethod
    def Einitial(x, alpha):
        return np.sin(math.pi * x) * alpha

    @staticmethod
    def generator(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0, N_inner, N_i, N_b, N_eta, N_E, N_Eb,
                  N_E0):
        # Sampling data
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

v_max = 3.5
V_T = 1
alpha= 0.2
t_max = 1.
pi = math.pi
x_max = pi
N_pred = 150
N_init = 150
N_E =10000
N_N = 100000
N_Eb = 2000
N_Einit = 2000
layers = [3,20,20,20,20,20,20,20,1]
layersE =[2,15,15,15,15,15,1]


x = np.linspace(-x_max,x_max,N_pred)
t = np.linspace(0,t_max,N_pred)
v = np.linspace(-v_max,v_max,N_pred)

T,X,V = np.meshgrid(t,x,v)
X_star = np.hstack((T.flatten()[:,None],X.flatten()[:,None],V.flatten()[:,None]))
lb = X_star.min(0)
ub = X_star.max(0)

print("lower_bound: ",lb)
print("upper_bound: ",ub)

xx2 = np.stack((T[0,:,:], X[0,:,:], V[0,:,:]),axis=2) #x=-pi condition
xx2 = np.reshape(xx2,(N_pred**2,3))
print("at -x_max position: ",xx2.shape)
xx3 = np.stack((T[-1,:,:],X[-1,:,:],V[-1,:,:]),axis=2)
xx3 = np.reshape(xx3,(N_pred**2,3))
print("at x_max position: ",xx3.shape)

xx4 = np.stack((T[:,:,0],X[:,:,0],V[:,:,0]),axis=2)
xx4 = np.reshape(xx4,(N_pred**2,3))
print("at -v_max position: ",xx4.shape)
#plt.plot(xx4[:,0],xx4[:,1],label='-x_max to x_max and time')
#plt.xlabel('time')
#plt.ylabel('-x_max to x_max position')
#plt.legend()
#plt.savefig('Vlasov_Equation.png')
#plt.show()

xx1 = np.stack((np.zeros(N_init**2),np.linspace(-x_max,x_max,N_init**2),np.clip(np.squeeze(np.random.multivariate_normal([0.],[[2.]],N_init**2)),-v_max,v_max)),axis=1)
ff1 = Vlasov1DNN.initialConditions(xx1[:,1],xx1[:,2],alpha,V_T)

X_f_train = np.vstack([xx1,xx2,xx3])

print("lowe_bound: ",lb)
print("upper_bound: ",ub)
print("upper - lower : ",ub-lb)
print("lb + (ub - lb): ",lb+(ub - lb))
print(X_f_train.shape)

X_N_train = lb + (ub - lb)* lhs(3,N_N)
X_N_train = np.vstack((X_N_train,X_f_train))

print(X_N_train.shape)

xbE = np.stack((np.linspace(0,t_max,N_Eb),-x_max*np.ones(N_Eb)),axis=1)
xtE = np.stack((np.linspace(0,t_max,N_Eb),x_max*np.ones(N_Eb)),axis=1)
print(xbE.shape)

X_E0 = np.stack((np.zeros(N_Einit),np.linspace(-x_max,x_max,N_Einit)),axis=1)
E_0 = Vlasov1DNN.Einitial(X_E0[:,1],alpha)
print("Lower_bounds: ",lb[0:2])
print("Upper_bounds: ",ub[0:2])
X_E = lb[0:2] + (ub[0:2] - lb[0:2])*lhs(2,N_E)
X_E = np.vstack((X_E,xbE,xtE))
print(X_E.shape)
X_E = np.stack((X_E[:,0],X_E[:,1],v_max*np.ones(N_E+2*N_Eb)),axis=1)
def generator(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
    # Example generator function that yields data batches
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


model = Vlasov1DNN(lb,ub,layers,layersE,X_N_train,X_E,xx1,ff1,xx2,xx3,xx4,xbE,xtE,X_E0,E_0,generator)
with tf.device('/device:GPU:0'):
    start_time = time.time()
    model.train(2, 64000, 16000, 16000, 0, 4000, 1000, 1000) # N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))