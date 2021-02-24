import math
import pennylane as qml


def variational_layer(params, wires):

    # Build entangling layer
    weights = params.reshape((-1, len(wires), 3))
    qml.templates.layers.StronglyEntanglingLayers(weights, wires=wires)


def embedding_layer(features, wires):

    # Build embedding layer
    qml.templates.embeddings.AngleEmbedding(features, wires=wires, rotation='Y')


@qml.template
def ttn_layer(params, input_size, bond_v):

    # Acquire sizes
    num_vqubits = int(input_size / bond_v)
    num_layers = int(math.log2(num_vqubits))

    # Add TTN layers
    param_index = 0
    for i in range(num_layers):
        for j in range(2 ** (num_layers - i - 1)):
            offset_1 = bond_v * (2 ** i) * (2 * j)
            offset_2 = bond_v * (2 ** i) * (2 * j + 1)
            wirelist = [offset_1 + k for k in range(bond_v)] + [offset_2 + k for k in range(bond_v)]
            variational_layer(params[param_index], wires=wirelist)
            param_index += 1


@qml.template
def mps_layer(params, input_size, bond_v):

    # Add MPS layers
    param_index = 0
    for i in reversed(range(0, input_size - bond_v, bond_v)):
        wirelist = [i + k for k in range(2 * bond_v)]
        variational_layer(params[param_index], wires=wirelist)
        param_index += 1
