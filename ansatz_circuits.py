import math
import ansatz_layers
import pennylane as qml


def ttn_ansatz_alpha(input_size, output_size, bond_v, device):

    # Acquire sizes
    num_vqubits = int(input_size / bond_v)
    num_layers = int(math.log2(num_vqubits))

    # Build core ansatz
    @qml.qnode(device)
    def var_ckt(features, params):
        qml.templates.embeddings.AngleEmbedding(features, wires=range(input_size), rotation='Y')
        ansatz_layers.ttn_layer(params, num_layers, bond_v)
        return qml.probs(range(output_size))

    # Return QNode
    return var_ckt


def mps_ansatz_alpha(input_size, output_size, bond_v, device):

    # Build core ansatz
    @qml.qnode(device)
    def var_ckt(features, params):
        qml.templates.embeddings.AngleEmbedding(features, wires=range(input_size), rotation='Y')
        ansatz_layers.mps_layer(params, input_size, bond_v)
        return qml.probs(range(output_size))

    # Return QNode
    return var_ckt
