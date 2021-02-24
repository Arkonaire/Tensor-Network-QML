import numpy as np
import pennylane as qml
import ansatz_circuits as anc
import matplotlib.pyplot as plt


class BarStripeGenerator:

    def __init__(self, shape, netstruct, device=None):

        # Initialize basic attributes
        self.shape = shape
        self.bond_v = netstruct['bond_v']
        self.network = netstruct['network']
        self.num_qubits = netstruct['num_qubits']
        self.num_vqubits = int(self.num_qubits / self.bond_v)
        self.num_layers = int(np.log2(self.num_vqubits))

        # Hyperparameters
        self.steps = 1000
        self.stepsize = 0.2

        # Load device and circuit
        self.device = device if device is not None else qml.device('default.qubit', wires=self.num_qubits)
        self.params = 2 * np.pi * np.random.rand(sum([2 ** i for i in range(self.num_layers)]), 6 * self.bond_v)
        self.var_ckt = anc.generative_ansatz(self.num_qubits, self.bond_v, self.device, network=self.network)

        # Build valid output set
        self.valid_set = np.zeros((2 ** self.shape[0] + 2 ** self.shape[1] - 2, self.shape[0] * self.shape[1]))
        self.taget_dist = np.zeros(self.num_qubits)
        self.build_valid_set()
        self.build_target_distribution()

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
            self.taget_dist[index] = 1 / self.valid_set.shape[0]

    def train_model(self):

        # Define cost function
        def costfunc(params):
            probs = self.var_ckt(params)
            loss = sum((probs - self.taget_dist) ** 2)
            return loss

        # Optimize
        opt = qml.MomentumOptimizer(stepsize=self.stepsize)
        for i in range(self.steps):
            self.params = opt.step(costfunc, self.params)

    def sample(self):

        # Sample output TODO: Visualize results
        probs = self.var_ckt(self.params)

    def show_grid(self, grid):

        # Plot data
        grid = grid.reshape(self.shape)
        plt.imshow(grid)
        plt.xticks(np.arange(0.5, self.shape[1], 1.0), [])
        plt.yticks(np.arange(-0.5, self.shape[0], 1.0), [])
        plt.grid(True)
        plt.show()
