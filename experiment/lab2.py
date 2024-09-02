import numpy as np
import tensorflow as tf



physical_gpus = tf.config.list_physical_devices('GPU')
physical_gpus


class PINN(tf.keras.Model):
    def __init__(self,layers,input,u_exact):
        self.layers = layers
        self.input = input
        self.x = input[:,0:1]
        self.t = input[:,1:2]
        self.u_exact = u_exact
        self.input_train= tf.concat([self.x,self.t],axis=1)
        self.model = self.Neural_net()


    def Neural_net(self):
        model = tf.keras.Sequential()
        for i in range(len(self.layers) - 1):
            model.add(tf.keras.layers.Dense(self.layers[i],activation='tanh'))
        model.add(tf.keras.layers.Dense(self.layers[-1]))
        return model
    def call(self):
        x= self.model(self.input_train)
        return x

    def loss_function(self):
        with tf.GradientTape(persistent=True) as tape:
            u_pred = self.call()
            u_t = tape.gradient(u_pred,self.t)
            u_x = tape.gradient(u_pred,self.x)
            u_xx = tape.gradient(u_x,self.x)
        alpha=0.01
        pde_loss = tf.reduce_mean(tf.square(u_t-alpha*u_xx))
        u_loss = tf.reduce_mean(tf.square(u_pred-self.u_exact))
        loss = u_loss+pde_loss
        return loss





np.random.seed(42)
num_samples =1000
x_train = np.random.uniform(low=0,high=1,size=(num_samples,1))
t_train = np.random.uniform(low=0,high=1,size=(num_samples,1))
u_exact = np.sin(np.pi*x_train)*np.exp(-np.pi**2*t_train)

x_train_tf = tf.convert_to_tensor(x_train,dtype=tf.float32)
t_train_tf = tf.convert_to_tensor(t_train,dtype=tf.float32)
u_exact_tf = tf.convert_to_tensor(u_exact,dtype=tf.float32)

input_train = tf.concat([x_train_tf,t_train_tf],axis=1)

x_test = np.linspace(0,1,100).reshape(-1,1)
t_test = np.linspace(0,1,100).reshape(-1,1)
u_exact_test = np.sin(np.pi * x_test) * np.exp(-np.pi ** 2 * t_test)

x_test_tf = tf.convert_to_tensor(x_test,dtype=tf.float32)
t_test_tf = tf.convert_to_tensor(t_test,dtype=tf.float32)
u_exact_test_tf = tf.convert_to_tensor(u_exact_test,dtype=tf.float32)

input_test = tf.concat([x_test,t_test],axis=1)

layers=[2,30,30,30,1]
pinn = PINN(layers=layers,input=input_train,u_exact=u_exact)
loss = pinn.loss_function()
print(loss)