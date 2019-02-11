from keras import Input, Model, optimizers, losses
from keras.layers import Dense


class drl_network():
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        # TODO probar capas mas pequeñas (n_hidden) y mas capas (x = Dense(n_hidden, activation='relu')(x))
        input1 = Input(shape=(n_input, ), name='input1')
        x = Dense(n_hidden, activation='relu')(input1)
        x = Dense(n_hidden, activation='relu')(x)
        out1 = Dense(n_output, activation='linear', name='out1')(x)
        self.model = Model(inputs=[input1], outputs=[out1])

        self.optimizer = optimizers.Adam(lr=learning_rate)
        self.loss = losses.mean_absolute_error
        #self.loss = losses.mean_squared_error

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
