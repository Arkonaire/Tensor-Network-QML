import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import ansatz_circuits as anc

import os.path
import pickle
import time


class RainForecast:

    def __init__(self, datapath, netstruct, device=None, entanglers=1):

        # Initialize basic attributes
        self.path = datapath
        self.bond_v = netstruct['bond_v']
        self.network = netstruct['network']
        self.input_size = netstruct['input_size']
        self.output_size = netstruct['output_size']
        self.num_vqubits = int(self.input_size / self.bond_v)
        self.num_layers = int(np.log2(self.num_vqubits))

        # Hyperparameters
        self.steps = 1000
        self.stepsize = 0.1
        self.batch_size = 10
        self.test_size = 1000
        self.cols_to_drop = [0, 4, 5, 6, 8, 9]
        self.cost_multiplier = 10

        # Load data
        with open(self.path, 'rb') as file:
            self.features, self.labels, self.data_stats = pickle.load(file)
            np.delete(self.features, self.cols_to_drop, axis=1)
            file.close()

        # Organize data
        self.train_X = self.features[:-self.test_size, :]
        self.test_X = self.features[-self.test_size:, :]
        self.train_Y = self.labels[:-self.test_size, :]
        self.test_Y = self.labels[-self.test_size:, :]

        # Load device and circuit
        self.device = device if device is not None else qml.device('default.qubit', wires=self.input_size)
        self.var_ckt = anc.discriminative_ansatz(self.input_size, self.output_size,
                                                 self.bond_v, self.device, network=self.network)

        # Initialize parameters
        param_size = sum([2 ** i for i in range(self.num_layers)]) if self.network == 'TTN' else self.num_vqubits - 1
        self.params = 2 * np.pi * np.random.rand(param_size, 6 * self.bond_v * entanglers)

        # Initialize output containers
        self.costs = np.zeros(self.steps)
        self.probs = None
        self.runtime = 0

    def train_model(self):

        # Define cost function TODO: Implement cost function
        def costfunc(params):

            # Initialization
            batch_index = np.random.randint(0, self.train_X.shape[0], (self.batch_size,))
            batch_X = self.train_X[batch_index]
            batch_Y = self.train_Y[batch_index]
            loss = 0

            # Calculate losses
            for i in range(self.batch_size):
                pred = self.var_ckt(batch_X[i], params)
                loss += (pred[0] - batch_Y[i][0]) ** 2 + (pred[1] - batch_Y[i][1]) ** 2

            # Return cost
            loss *= self.cost_multiplier
            return loss

        # Prepare to optimize
        opt = qml.MomentumOptimizer(stepsize=self.stepsize)
        self.costs = np.zeros(self.steps)
        t0 = time.time()

        # Optimize
        for i in range(self.steps):
            self.params, self.costs[i] = opt.step_and_cost(costfunc, self.params)
            if (i + 1) % 100 == 0 or i == 0:
                print('Iteration ' + str(i + 1) + ': Cost = ' + str(self.costs[i]))

        # Record runtime
        self.runtime = time.time() - t0
        print('Execution Time: ' + str(self.runtime) + ' sec')

        # Acquire valid filename for save
        filename = self.network + '_0'
        filename = 'discriminative_models/' + filename
        file_index = 0
        while os.path.exists(filename):
            file_index += 1
            filename = filename[:-1] + str(file_index)

        # Save data
        with open(filename, 'wb') as file:
            pickle.dump([self.costs, self.params, self.runtime], file)
            file.close()

    def load_model(self, file_index=0):

        # Load trained parameters
        filename = self.network + '_' + str(file_index)
        filename = 'discriminative_models/' + filename
        with open(filename, 'rb') as file:
            self.costs, self.params, self.runtime = pickle.load(file)
            file.close()

    def sample(self, features, params=None):

        # Sample output
        params = self.params if params is None else params
        self.probs = self.var_ckt(features, params)

    def visualize(self):

        # Sample if required
        if self.probs is None:
            self.sample(self.test_X[np.random.randint(0, self.test_size)])

        # Print output
        index_max = max(range(len(self.probs)), key=self.probs.__getitem__)
        bits = format(index_max, '0' + str(self.output_size) + 'b')
        bits = np.array([int(i) for i in bits])
        rain_today = (bits[0] == 1)
        rain_tomorrow = (bits[1] == 1)
        print('Rain Today?: ', rain_today)
        print('Rain Tomorrow?: ', rain_tomorrow)

        # Plot probability distribution
        plt.figure()
        plt.bar(range(len(self.probs)), self.probs)
        plt.title('Output distribution')
        plt.xlabel('Sample Output')
        plt.ylabel('Probability')
        plt.grid(True)

        # Plot cost curve
        plt.figure()
        plt.plot(range(len(self.costs)), self.costs)
        plt.title('Cost Curve')
        plt.xlabel('Iteration No.')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
