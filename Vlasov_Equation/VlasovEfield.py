import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from pyDOE import lhs
np.random.seed(42)
tf.random.set_seed(42)

class vlasov1DNN:
    def __init__(self,lb,ub,layers,layersE , X_inner,E_inner, X_i , f_i , x_b, x_t,x_eta, E_b , E_t, X_E0 ,E_0,generator):
        self.lb=lb
        self.ub=ub
        self.layers=layers
        self.layersE = layersE
        self.lbE= lb[0:2]
        self.ubE= ub[0:2]
        self.generator = generator
        self.iters=0


        self.weights, self.biases = self.Initialize_NN(self.layers)
        self.Eweights, self.Ebiases = self.Initialize_NN(self.layersE)

        with tf.name_scope('Inputs'):
            # N_inner tells us where to evaluate the function on the inside of the domain. It
            #is a N by 2 tensor where the 0th index is t and the 1st index is x
            self.N_t = X_inner[:,0]
            self.N_x = X_inner[:,1]
            self.N_v = X_inner[:,2]
            self.X_inner = X_inner

            # E_inner tells us where to evaluate the electric field equation (Gauss's law) on
            # the interior of the domain
            self.E_inner = E_inner

            self.E_b = E_b
            self.E_t = E_t

            #initial_condition
            self.f_i_t = X_i[:,0]
            self.f_i_x = X_i[:,1]
            self.f_i_v = X_i[:,2]
            self.f_i = f_i
            self.X_i = X_i
            self.X_E0 = X_E0
            self.E_0 = E_0


    def Initialize_NN(self, layers):
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            W = self.Xavier_init(layers[i],layers[i+1])
            b = tf.Variable(tf.zeros([1,layers[i+1]],dtype=tf.float32),dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def Xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(6/(in_dim+out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim,out_dim],stddev=xavier_stddev),dtype=tf.float32)

    def initialConditions(X, V, alpha, V_T):
        return (np.exp(-(V ** 2 / (V_T ** 2))) / (math.sqrt(math.pi) * V_T)) * (1 + alpha * np.sin(2 * X))

    def Einitial(X, alpha):
        return 4 * math.pi * alpha * np.cos(2 * X) / 2.

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