import tflearn
import numpy as np
import sys
import os


class Generalizer(object):

    def __init__(self, savedModel=None):

        tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

        net = tflearn.input_data(shape=[None, 7])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 15, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',
                                 batch_size=50, learning_rate=0.01)

        self.model = tflearn.DNN(net)

        if savedModel is not None:
            self.model.load(savedModel)

    def train(self, inputs, y):
        self.model.fit(inputs, y, n_epoch=10, validation_set=(inputs, y))

    def getAction(self, state):
        return np.argmax(self.model.predict(state))


if __name__ == "__main__":
    data = np.load(sys.argv[1])

    dataId = ''.join(os.path.basename(sys.argv[1]).split('.'))

    inputs = np.copy(data['inputs'])
    inputs -= np.mean(inputs, axis=0)
    y = np.copy(data['y'])

    gen = Generalizer()
    # gen.train(inputs, y)

    # gen.model.save(dataId + '_weights')
