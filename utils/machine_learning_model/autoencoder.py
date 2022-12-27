from keras.layers import Dense, Input
from keras.models import Model, load_model
import pprint as pp

class Autoencoder:
    def __init__(self, input_dim: int, hidden_dim: list, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.build_model()

    def build_model(self):
        self.input_layer = Input(shape=(self.input_dim,))
        self.encoder = Dense(self.hidden_dim[0], activation='relu')(self.input_layer)
        for dim in self.hidden_dim[1:]:
            self.encoder = Dense(dim, activation='relu')(self.encoder)
        self.decoder = Dense(self.output_dim, activation='sigmoid')(self.encoder)
        self.model = Model(self.input_layer, self.decoder)

        self.model.compile(optimizer='adam', loss='mse')

    def train(self, x_train, epochs):
        self.model.fit(x_train, x_train, epochs=epochs)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def get_encoder(self):
        return Model(self.input_layer, self.encoder)

    def get_decoder(self):
        return Model(self.encoder, self.decoder)


if __name__ == '__main__':
    ae = Autoencoder(150, [100, 50], 150)