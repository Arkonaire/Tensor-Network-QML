import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import ansatz_circuits as anc

import os.path
import pickle
import time


class RainForecast:

    def __init__(self, shape, netstruct, device=None, entanglers=1):

        # Initialize basic attributes
        self.shape = shape
        self.bond_v = netstruct['bond_v']
        self.network = netstruct['network']
        self.input_size = netstruct['input_size']
        self.output_size = netstruct['output_size']
        self.num_vqubits = int(self.input_size / self.bond_v)
        self.num_layers = int(np.log2(self.num_vqubits))

        # Hyperparameters
        self.steps = 1000
        self.stepsize = 0.1
        self.cost_multiplier = 10

        # Load data
        with open('rain_dataset/processed_data') as file:
            self.data = pickle.load(file)
            file.close()

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
            probs = self.var_ckt(params)
            loss = sum(probs ** 2)
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
        filename = self.network + '_' + str(self.shape[0]) + '_' + str(self.shape[1]) + '_0'
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
        filename = self.network + '_' + str(self.shape[0]) + '_' + str(self.shape[1])
        filename = 'discriminative_models/' + filename + '_' + str(file_index)
        with open(filename, 'rb') as file:
            self.costs, self.params, self.runtime = pickle.load(file)
            file.close()

    def sample(self, params=None):

        # Sample output
        params = self.params if params is None else params
        self.probs = self.var_ckt(params)

    def visualize(self):

        # Sample if required
        if self.probs is None:
            self.sample()

        # Print output
        index_max = max(range(len(self.probs)), key=self.probs.__getitem__)
        bits = format(index_max, '0' + str(self.output_size) + 'b')
        bits = np.array([int(i) for i in bits])
        print('Output = ', bits)     # TODO: More infromative output

        # Plot probability distribution TODO: Deal with the resolution problem.
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
