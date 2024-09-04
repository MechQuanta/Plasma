import tensorflow as tf
import numpy as np


class vlasov1DNN:

    def __init__(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0,
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

        # Define model inputs
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

        # Build the model
        self.model = self.build_model()

        # Define the optimizer
        self.optimizer = tf.keras.optimizers.Adam()

    def build_model(self):
        inputs = tf.keras.Input(shape=(3,))  # Shape (t, x, v)

        # Define the neural network structure
        x = inputs
        for layer in self.layers:
            x = tf.keras.layers.Dense(layer, activation='tanh')(x)

        N_output = tf.keras.layers.Dense(1, name='N_output')(x)

        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=N_output)

        return model

    def custom_loss(self, y_true, y_pred):
        N_pred = y_pred
        NLoss = tf.reduce_mean(tf.square(N_pred))
        # Define other losses here
        # self.gaussLoss, self.iLoss, etc.
        loss = NLoss  # Update with all losses
        return loss

    def train(self, numSteps, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
        for step in range(numSteps):
            for (X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0) in self.generator(
                    self.X_inner, self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta, self.E_inner, self.E_b, self.E_t,
                    self.X_E0, self.E_0,
                    N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
                with tf.GradientTape() as tape:
                    # Forward pass
                    N_pred = self.model(X_inner)
                    loss = self.custom_loss(None, N_pred)  # Update with actual inputs and predictions

                # Compute gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                print(f'Step: {step}, Loss: {loss.numpy()}')

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
        # Xavier initialization
        in_dim, out_dim = size
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.random.normal(shape=[in_dim, out_dim], mean=0.0, stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)
