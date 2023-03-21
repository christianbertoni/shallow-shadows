from tensor_networks import *
from scipy.stats import unitary_group
from qiskit.quantum_info import random_clifford
from numpy import array, transpose, kron, identity, shape, empty
from numpy.random import random
import sys
import h5py

# CODE FOR DATA ACQUISITION SIMULATION

# INPUT:
# state: MPS; an MPS corresponding to the state
# depth: integer; desired circuit depth
# number_of_snapshots: integer; desired number of snapshots in the shadow
# output_file_name: desired name for the file in which the shadow is saved
# clifford: bool; if True, clifford unitaries will be sampled as the local gates in the circuit. If False, Haar random unitaries will be sampled instead
# verbose: if True, the number of snapshot generated will be displayed

# OUTPUT: an H5PY file with one dataset containing all snapshots in the form of matrix product states, the tensors follow the same convention as in tensor_networks.py


def generateShadow(state, depth, number_of_snapshots, output_file_name, clifford=True, verbose=False):
    size = state.getSize()
    bond_dimension = max(1, 2**(depth-1))
    output_file = h5py.File(output_file_name, 'w')
    output_file.create_dataset('snapshots', (number_of_snapshots,
                               size, bond_dimension, bond_dimension, 4), dtype=complex)

    for k in range(number_of_snapshots):

        # construct an MPO corresponding to a random circuit of the required size and depth
        circuit = randomCircuitMPO(size, depth, clifford)

        rotated_state = MPOtimesMPS(circuit, state)

        # the measurement result is a list of bits, 0 or 1 for each qubit
        measurement_result = sampleFromMPS(rotated_state)

        # MPS tensors for the classical snapshot U^dagger|b>
        classical_snapshot_tensors = []

        for i in range(size):

            local_tensors = empty(
                (bond_dimension, bond_dimension, 4), dtype=complex)

            for j in range(bond_dimension):
                for l in range(bond_dimension):
                    local_tensors[j][l] = transpose(conj(circuit.getMatrix(i, j, l)))[
                        :, measurement_result[i]]

            classical_snapshot_tensors.append(local_tensors)

        output_file["snapshots"][k] = classical_snapshot_tensors

        if (verbose):
            print('\r'+"Generated "+str(k+1) + " out of " +
                  str(number_of_snapshots)+" snapshots.", end='')

    if (verbose):
        print('\r'+"Generated "+str(number_of_snapshots) + " out of " +
              str(number_of_snapshots)+" snapshots.", end='')

    output_file.close()


# generates a random circuit and writes it as an MPO

def randomCircuitMPO(size, depth, clifford=False):
    bond_dimension = max(1, 2**(depth-1))
    MPO_tensors = []

    if (depth == 0):
        for i in range(size):
            if (clifford):
                U1 = random_clifford(1).to_matrix()
                U2 = random_clifford(1).to_matrix()
            else:
                U1 = array(unitary_group.rvs(2))
                U2 = array(unitary_group.rvs(2))
            MPO_tensors.append([[kron(U1, U2)]])

    elif (depth == 1):
        for i in range(size):
            if (clifford):
                U = random_clifford(2).to_matrix()
            else:
                U = array(unitary_group.rvs(4))
            MPO_tensors.append([[U]])

    else:
        for i in range(size):
            MPO_tensor = empty(
                (bond_dimension, bond_dimension, 4, 4), dtype=complex)

            if (clifford):
                unitary_list = [random_clifford(
                    2).to_matrix() for k in range(depth)]
            else:
                unitary_list = [array(unitary_group.rvs(4))
                                for k in range(depth)]

            for j in range(bond_dimension):
                for k in range(bond_dimension):
                    j_bin = int2list(j, depth-1)
                    k_bin = int2list(k, depth-1)
                    T = unitary_list[0]@transpose(
                        kron(identity(2)[j_bin[0]], identity(2)))
                    for r in range(1, depth-1):
                        T = T @ kron(identity(2), identity(2)[k_bin[r-1]])@unitary_list[r]@transpose(
                            kron(identity(2)[j_bin[r]], identity(2)))
                    MPO_tensor[j, k] = T @ kron(identity(2), identity(2)
                                                [k_bin[depth-2]])@unitary_list[depth-1]
            MPO_tensors.append(MPO_tensor)

    return MPO(tensors=MPO_tensors)


# given an MPS, simulates measurement in the computational basis. Outputs a string of bits corresponding to the measurement result.

def sampleFromMPS(M):
    bits = []
    normalization = 1.
    for site in range(M.getSize()):
        transfer_matrix = identity(M.getBondDimension()**2)
        for k in range(site):
            transfer_matrix = transfer_matrix@kron(
                M.bondMatrix(k, bits[k]), conj(M.bondMatrix(k, bits[k])))
        for k in range(M.getSize()-1, site, -1):
            transfer_matrix = sum(kron(M.bondMatrix(k, b), conj(M.bondMatrix(
                k, b))) for b in range(M.getPhysicalDimension()))@transfer_matrix

        probabilities = []
        for bit in range(M.getPhysicalDimension()):
            probabilities.append(abs(trace(
                transfer_matrix@kron(M.bondMatrix(site, bit), conj(M.bondMatrix(site, bit))))/normalization))
        chosen_bit = sampleFromDistribution(probabilities)
        normalization *= probabilities[chosen_bit]
        bits.append(chosen_bit)
    return bits


def sampleFromDistribution(probabilities):
    size = len(probabilities)
    cumulative_distribution = [
        0]+[sum(probabilities[:k]) for k in range(1, size+1)]
    r = random()
    for k in range(size):
        if (r >= cumulative_distribution[k] and r <= cumulative_distribution[k+1]):
            return k
