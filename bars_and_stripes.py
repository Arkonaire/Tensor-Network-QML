import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import ansatz_circuits as anc
import os.path
import pickle
import time


class BarStripeGenerator:

    def __init__(self, shape, netstruct, device=None, entanglers=1):

        # Initialize basic attributes
        self.shape = shape
        self.bond_v = netstruct['bond_v']
        self.network = netstruct['network']
        self.num_qubits = netstruct['num_qubits']
        self.num_vqubits = int(self.num_qubits / self.bond_v)
        self.num_layers = int(np.log2(self.num_vqubits))

        # Hyperparameters
        self.steps = 1000
        self.stepsize = 0.1
        self.cost_multiplier = 10

        # Load device and circuit
        self.device = device if device is not None else qml.device('default.qubit', wires=self.num_qubits)
        self.var_ckt = anc.generative_ansatz(self.num_qubits, self.bond_v, self.device, network=self.network)

        # Build valid output set
        self.valid_set = np.zeros((2 ** self.shape[0] + 2 ** self.shape[1] - 2, self.shape[0] * self.shape[1]))
        self.valid_set = self.valid_set.astype(int)
        self.target_dist = np.zeros(2 ** self.num_qubits)
        self.build_valid_set()
        self.build_target_distribution()

        # Initialize parameters
        self.params = 2 * np.pi * np.random.rand(sum([2 ** i for i in range(self.num_layers)]),
                                                 6 * self.bond_v * entanglers)

        # Initialize output containers
        self.costs = np.zeros(self.steps)
        self.probs = None
        self.runtime = 0

    def build_valid_set(self):

        # Build bars
        index = 0
        for z in range(2 ** self.shape[0]):
            grid = np.zeros(self.shape)
            bits = format(z, '0' + str(self.shape[0]) + 'b')
            for i in range(self.shape[0]):
                if bits[i] == '1':
                    grid[i, :] = 1
            self.valid_set[index, :] = grid.reshape(-1)
            index += 1

        # Build stripes
        for z in range(1, 2 ** self.shape[1] - 1):
            grid = np.zeros(self.shape)
            bits = format(z, '0' + str(self.shape[1]) + 'b')
            for i in range(self.shape[1]):
                if bits[i] == '1':
                    grid[:, i] = 1
            self.valid_set[index, :] = grid.reshape(-1)
            index += 1

    def build_target_distribution(self):

        # Build target probability distribution
        for z in self.valid_set:
            bitstr = "".join([str(i) for i in z])
            index = int(bitstr, 2)
            self.target_dist[index] = 1 / self.valid_set.shape[0]

    def train_model(self):

        # Define cost function
        def costfunc(params):
            probs = self.var_ckt(params)
            loss = sum((probs - self.target_dist) ** 2)
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
        filename = 'models/' + filename
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
        filename = 'models/' + filename + '_' + str(file_index)
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

        # Plot output
        plt.figure()
        index_max = max(range(len(self.probs)), key=self.probs.__getitem__)
        bits = format(index_max, '0' + str(self.num_qubits) + 'b')
        grid = np.array([int(i) for i in bits])
        self.show_grid(grid)

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

    def show_grid(self, grid):

        # Plot data
        grid = grid.reshape(self.shape)
        plt.imshow(grid)
        plt.xticks(np.arange(0.5, self.shape[1], 1.0), [])
        plt.yticks(np.arange(-0.5, self.shape[0], 1.0), [])
        plt.grid(True)
        plt.show()
