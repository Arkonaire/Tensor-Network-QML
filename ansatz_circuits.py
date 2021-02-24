import ansatz_layers
import pennylane as qml


def discriminative_ansatz(input_size, output_size, bond_v, device, network='TTN'):

    # Select tensor network layer
    if network == 'TTN':
        variational_layer = ansatz_layers.ttn_layer
    elif network == 'MPS':
        variational_layer = ansatz_layers.mps_layer
    else:
        raise KeyError('Invalid Ansatz')

    # Build core ansatz
    @qml.qnode(device)
    def var_ckt(features, params):
        ansatz_layers.embedding_layer(features, wires=range(input_size))
        variational_layer(params, input_size, bond_v)
        return qml.probs(range(output_size))

    # Return QNode
    return var_ckt


def generative_ansatz(num_qubits, bond_v, device, network='TTN'):

    # Select tensor network layer
    if network == 'TTN':
        layer = ansatz_layers.ttn_layer
    elif network == 'MPS':
        layer = ansatz_layers.mps_layer
    else:
        raise KeyError('Invalid Ansatz')

    # Build core ansatz
    @qml.qnode(device)
    def var_ckt(params):
        qml.inv(layer(params, num_qubits, bond_v))
        return qml.probs(range(num_qubits))

    # Return QNode
    return var_ckt
