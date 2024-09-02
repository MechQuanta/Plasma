import tensorflow as tf
import numpy as np

class PINN():
    def __init__(self,x,t,layers,lr=0.001,**kwargs):


        self.x_train_tf =x
        self.t_train_tf =t




    def structure(self,layers):
        model= tf.keras.Sequential()
        for layer in layers[0:-1]:
            model.add(tf.keras.layers.Dense(layer,activation='tanh'))
        model.add(tf.keras.layers.Dense(layers[-1],activation=None))
        return model
    def optimizers(self,learning_rate=0.001):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self,inputs):
        x = inputs[:,0:1]
        t = inputs[:,1:2]
        concat_input = tf.concat([x,t],axis=1)
        output = self.structure(concat_input)
        return output
    def loss_function(self,input,u_exact):
        u_pred = self.call(input)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_train_tf)
            tape.watch(t_train_tf)
            u_t = tape.gradient(u_pred,t_train_tf)
            u_x = tape.gradient(u_pred,x_train_tf)
            u_xx = tape.gradient(u_x,x_train_tf)
            del tape
        alpha = 0.01
        pde_loss = tf.reduce_mean(tf.square(u_t - alpha*u_xx))
        u_loss = tf.reduce_mean(tf.square(u_pred-u_exact))
        loss = pde_loss+u_loss
        return loss
    def train_step(self,input,u_exact):
        with tf.GradientTape() as tape:
            loss = self.loss_function(input,u_exact)
        self.optimizers = self.optimizers(learning_rate=lr)
        loss_grad = tape.gradient(loss,self.nn.trainable_variables)
        self.optimizers.apply_gradients(zip(loss_grad,self.nn.trainable_variables))
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

num_epochs = 1000
pinn = PINN(x_train_tf,t_train_tf,layers=[2,50,50,50,1],lr=0.01)
for epoch in range(num_epochs):
    loss = pinn.train_step(input_train,u_exact_tf)
    if epoch % 10 == 0:
        print(f'Epoch:{epoch} , loss : {loss.numpy}')










