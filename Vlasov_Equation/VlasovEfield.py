import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy
import scipy.io

import time

np.random.seed(42)
tf.random.set_seed(42)

class vlasov1DNN:
    def __init__(self,lb,ub,layers,layersE , X_inner,E_inner, X_i , f_i , X_b, X_t,X_eta, E_b , E_t, X_E0 ,E_0,generator):
        self.lb=lb
        self.ub=ub
        self.layers=layers
        self.layersE = layersE
        self.lbE= lb[0:2]
        self.ubE= ub[0:2]
        self.generator = generator
        self.iters=0


        self.weights, self.biases = self.Initialize_NN(self.layers)
        self.Eweights, self.Ebiases = self.Initialize_NN_E(self.layersE)

        with tf.name_scope('Inputs'):
            # N_inner tells us where to evaluate the function on the inside of the domain. It
            # is a N by 2 tensor where the 0th index is t and the 1st index is x
            self.N_t = X_inner[:, 0]
            self.N_x = X_inner[:, 1]
            self.N_v = X_inner[:, 2]
            self.X_inner = X_inner

            # E_inner tells us where to evaluate the electric field equation (Gauss's law) on
            # the interior of the domain
            self.E_inner = E_inner

            self.E_b = E_b
            self.E_t = E_t

            # Initial conditions
            self.f_i_t = X_i[:, 0]
            self.f_i_x = X_i[:, 1]
            self.f_i_v = X_i[:, 2]
            self.f_i = f_i
            self.X_i = X_i
            self.X_E0 = X_E0
            self.E_0 = E_0

            # Boundary conditions
            self.f_t_t = X_t[:, 0]
            self.f_t_x = X_t[:, 1]
            self.f_t_v = X_t[:, 2]
            self.X_t = X_t
            self.f_b_t = X_b[:, 0]
            self.f_b_x = X_b[:, 1]
            self.f_b_v = X_b[:, 2]
            self.X_b = X_b

            # Eta evaluated
            self.X_eta = X_eta
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        with tf.name_scope("Placeholders"):
            self.N__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.N__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.N__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_i__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_i__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_i__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f__i = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_b__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_b__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_b__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_t__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_t__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.f_t__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.eta__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.eta__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.eta__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E__v = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_b__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_b__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_t__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_t__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_i__t = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E_i__x = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.E__0 = tf.compat.v1.placeholder(tf.float32, shape=[None])


        self.N_pred = self.net_N(self.N__t, self.N__x, self.N__v)
        self.f_i_pred = self.net_f(self.f_i__t, self.f_i__x, self.f_i__v)
        self.f_b_pred = self.net_f(self.f_b__t, self.f_b__x, self.f_b__v)
        self.f_t_pred = self.net_f(self.f_t__t, self.f_t__x, self.f_t__v)
        self.f_db_pred = self.net_f_d(self.f_b__t, self.f_b__x, self.f_b__v)
        self.f_dt_pred = self.net_f_d(self.f_t__t, self.f_t__x, self.f_t__v)
        self.f_d_pred = self.net_f_d(self.f_i__t, self.f_i__x, self.f_i__v)
        self.eta_pred = self.net_eta(self.eta__t, self.eta__x, self.eta__v)
        self.eta_t_pred = self.net_eta(self.f_t__t, self.f_t__x, self.f_t__v)
        self.eta_b_pred = self.net_eta(self.f_b__t, self.f_b__x, self.f_b__v)
        self.gauss_pred = self.net_gauss(self.E__t, self.E__x, self.E__v)
        self.phi_b_pred = self.net_phi(self.E_b__t, self.E_b__x)
        self.phi_t_pred = self.net_phi(self.E_t__t, self.E_t__x)
        self.E_b_pred = self.net_E(self.E_b__t, self.E_b__x)
        self.E_t_pred = self.net_E(self.E_t__t, self.E_t__x)
        self.E_db_pred = self.net_E_d(self.E_b__t, self.E_b__x)
        self.E_dt_pred = self.net_E_d(self.E_t__t, self.E_t__x)
        self.E_i_pred = self.net_E(self.E_i__t, self.E_i__x)

        self.NLoss = tf.reduce_mean(tf.square(self.N_pred))
        self.gaussLoss = tf.reduce_mean(tf.square(self.gauss_pred))
        self.iLoss = tf.reduce_mean(tf.square(self.f_i_pred - self.f__i))
        self.bLoss = tf.reduce_mean(tf.square(self.f_t_pred - self.f_b_pred)) + \
                     tf.reduce_mean(tf.square(self.f_dt_pred - self.f_db_pred))
        # .5*tf.reduce_mean(tf.square(self.eta_t_pred - self.eta_b_pred))
        # self.etaLoss = tf.reduce_mean(tf.square(self.eta_pred))
        self.EbLoss = tf.reduce_mean(tf.square(self.E_b_pred - self.E_t_pred)) + \
                      tf.reduce_mean(tf.square(self.E_db_pred - self.E_dt_pred)) + \
                      tf.reduce_mean(tf.square(self.phi_b_pred - self.phi_t_pred))
        self.E0loss = tf.reduce_mean(tf.square(self.E__0 - self.E_i_pred))
        self.loss = self.iLoss + self.bLoss + self.NLoss + \
                    self.gaussLoss + self.EbLoss + self.E0loss  # + self.etaLoss

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 1000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)


    def Initialize_NN(self, layers):
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            W = self.Xavier_init(layers[i],layers[i+1])
            b = tf.Variable(tf.zeros([1,layers[i+1]],dtype=tf.float32),dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def Initialize_NN_E(self,layersE):
        weights = []
        biases = []
        for i in range(len(layersE) - 1):
            W = self.Xavier_init(layersE[i],layersE[i+1])
            b = tf.Variable(tf.zeros([1,layersE[i+1]],dtype=tf.float32),dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    def forward(self,X,weights,biases,lb,ub):
        num_layers =len(weights)+1
        H = 2.0 * (X - lb)/(ub-lb) + 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H,W),b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H,W),b)
        return Y
    def net_eta(self,t,x,v):
        eta = self.forward(tf.stack([t,x,v],axis=1),self.weights,self.biases,self.lb,self.ub)
        return tf.squeeze(eta)
    def net_phi(self,t,x):
        phi = self.forward(tf.stack([t,x],axis=1),self.Eweights,self.Ebiases,self.lbE,self.ubE)
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


    def callback(self,loss,NLoss,iLoss, bLoss, gaussLoss, EbLoss, E0Loss):
        self.iters+=1
        if self.iters % 10 == 0:
            print('Loss: {}'.format(loss))
            print('Interior loss: {} Initial loss: {} Boundarty loss: {}'.format(NLoss,iLoss,bLoss))
            print('E0 loss: {} Gauss\'s loss: {} E boundary loss: {}'.format(E0Loss,gaussLoss,EbLoss))

    def train(self,numSteps,N_inner,N_i,N_b,N_eta,N_E, N_Eb,N_E0):
        for i in range(numSteps):
            for X_inner,X_i,f_i,X_b,X_t,X_eta,E_inner,E_b,E_t,X_E0,E_0 in self.generator(self.X_inner,self.X_i,self.f_i,self.X_b,self.X_t,self.X_eta,self.E_inner,self.E_b,self.E_t,self.X_E0,self.E_0,N_inner,N_i,N_b,N_eta,N_E,N_Eb,N_E0):
                tf_dict = {self.N__t:X_inner[:,0],self.N__x:X_inner[:,1],self.N__v:X_inner[:,2],self.f_i__t:X_i[:,0],self.f_i__x:X_i[:,1],self.f_i__v:X_i[:,2],self.f__i:f_i,self.f_b__t:X_b[:,0],self.f_b__x:X_b[:,1],self.f_b__v:X_b[:,2],self.f_t__t:X_t[:,0],self.f_t__x:X_t[:,1],self.f_t__v:X_t[:,2],self.eta__t:X_eta[:,0],self.eta__x:X_eta[:,1],self.eta__v:X_eta[:,2],self.E__t:E_inner[:,0],self.E__x:E_inner[:,1],self.E__v:E_inner[:,2],self.E_b__t:E_b[:,0],self.E_b__x:E_b[:,1],self.E_t__t:E_t[:,0],self.E_t__x:E_t[:,1],self.E_i__t:X_E0[:,0],self.E_i__x:X_E0[:,1],self.E__0:E_0}
                self.optimizer.minimize(self.sess,feed_dict=tf_dict,fetches=[self.loss,self.NLoss,self.iLoss,self.bLoss,self.gaussLoss,self.EbLoss,self.E0loss],loss_callback=self.callback)

    def Xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(6/(in_dim+out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim,out_dim],stddev=xavier_stddev),dtype=tf.float32)

    def initialConditions(X, V, alpha, V_T):
        return (np.exp(-(V ** 2 / (V_T ** 2))) / (math.sqrt(math.pi) * V_T)) * (1 + alpha * np.sin(2 * X))

    def Einitial(X, alpha):
        return 4 * math.pi * alpha * np.cos(2 * X) / 2.
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
        return (X_inner,X_i,f_i,X_b,X_t,X_eta,E_inner,E_b,E_t,X_E0,E_0)

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
ff1 = vlasov1DNN.initialConditions(xx1[:,1],xx1[:,2],alpha,V_T)

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
E_0 = vlasov1DNN.Einitial(X_E0[:,1],alpha)
print("Lower_bounds: ",lb[0:2])
print("Upper_bounds: ",ub[0:2])
X_E = lb[0:2] + (ub[0:2] - lb[0:2])*lhs(2,N_E)
X_E = np.vstack((X_E,xbE,xtE))
print(X_E.shape)
X_E = np.stack((X_E[:,0],X_E[:,1],v_max*np.ones(N_E+2*N_Eb)),axis=1)
model = vlasov1DNN(lb,ub,layers,layersE,X_N_train,X_E,xx1,ff1,xx2,xx3,xx4,xbE,xtE,X_E0,E_0,generator)