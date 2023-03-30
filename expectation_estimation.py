from tensor_networks import *
from inverse_MPS import measurementMap
import h5py
import numpy as np
import sys
from copy import deepcopy


# EXPECTATION ESTIMATION

# can be done for either shallow or sparse observables.

# SHALLOW ESTIMATION:
# inputs:
# shadow_file_name: string; name of the file where the classical shadow is stored
# inverse_file_name: string; name of the file where the inverse MPS is stored
# observable: MPS; vectorized version of the observable MPO
# output_file_name: string; name of the file where the estimations will be stored
# verbose: bool; if True, the number of estimations generated until now will be displayed


# SPARSE ESTIMATION:
# inputs:
# shadow_file_name: string; name of the file where the classical shadow is stored
# depth: integer; depth of the circuits used to generate the shadows
# paulis: list; list of paulis in the decomposition of the observable. A pauli is represented as a list of the single qubit paulis
#       where I=0, X=1, Y=2, Z=3, e.g. XXZIIYI=[1,1,3,0,0,2,0]
# coefficients: list; list of coefficients in front of each pauli in the decomposition
# output_file_name: string; name of the file where the estimations will be stored
# verbose: bool; if True, the number of estimations generated until now will be displayed

def shallowEstimation(shadow_file_name, inverse_file_name, observable, output_file_name, verbose=False):
    inverse_file = h5py.File(inverse_file_name, 'r')
    M_inverse = MinverseMPO(inverse_file['inverse_MPS'][:])
    inverse_file.close()
    inverted_observable = MPOtimesMPS(M_inverse, observable)

    shadow_file = h5py.File(shadow_file_name, 'r')
    output_file = h5py.File(output_file_name, 'w')
    number_of_snapshots = len(shadow_file['snapshots'][:])
    output_file.create_dataset(
        "estimations", (number_of_snapshots,), dtype=complex)

    for i, s in enumerate(shadow_file['snapshots'][:]):
        s_MPS = MPS(tensors=s)
        s_tensor2 = MPStensorProduct(s_MPS, s_MPS)
        output_file["estimations"][i] = MPSinnerProduct(
            s_tensor2, inverted_observable)
        if (verbose):
            print('\r'+"Generated "+str(i+1) + " out of " +
                  str(number_of_snapshots)+" estimations.", end='')

    if (verbose):
        print('\r'+"Generated "+str(number_of_snapshots) + " out of " +
              str(number_of_snapshots)+" estimations.", end='')

    output_file.close()
    shadow_file.close()


def sparseEstimation(shadow_file_name, depth, paulis, coefficients, output_file_name, verbose=False):
    I = [[1, 0], [0, 1]]
    X = [[0, 1], [1, 0]]
    Y = [[0, -1j], [1j, 0]]
    Z = [[1, 0], [0, -1]]

    pauli_list = [I, X, Y, Z]
    n_qubits = len(paulis[0])
    if (depth > 0):
        M = measurementMap(n_qubits//2, depth)

    paired_paulis = []
    non_zero_pairs = []
    eigenvalues = []
    for pauli in paulis:
        if (depth == 0):
            eigenvalues.append(1/3**sum([1 for x in pauli if x != 0]))
        else:
            eigenvalues.append(M.vectorElement(sum(
                2**(n_qubits//2-k)*int(pauli[2*k]+pauli[2*k+1] != 0) for k in range(n_qubits//2))))

        paired_paulis.append([])
        non_zero_pairs.append([])

        for k in range(0, n_qubits, 2):
            if (pauli[k]+pauli[k+1] > 0):
                non_zero_pairs[-1].append(k//2)
                paired_paulis[-1].append(np.kron(pauli_list[pauli[k]],
                                         pauli_list[pauli[k+1]]))
    shadow_file = h5py.File(shadow_file_name, 'r')
    output_file = h5py.File(output_file_name, 'w')
    number_of_shadows = len(shadow_file['snapshots'][:])

    output_file.create_dataset(
        "estimations", (number_of_shadows,), dtype=complex)

    for i, s in enumerate(shadow_file['snapshots'][:]):
        shadow_MPS = MPS(tensors=s)
        estimation = 0
        for k in range(len(paulis)):
            shadow_MPS_copy = deepcopy(shadow_MPS)
            for j in range(len(non_zero_pairs[k])):
                shadow_MPS_copy.multiplyByMatrix(
                    non_zero_pairs[k][j], paired_paulis[k][j])

            estimation += MPSinnerProduct(shadow_MPS,
                                          shadow_MPS_copy)/eigenvalues[k]*coefficients[k]

        output_file["estimations"][i] = estimation
        if (verbose):
            print('\r'+"Generated "+str(i+1) + " out of " +
                  str(number_of_shadows)+" estimations.", end='')

    if (verbose):
        print('\r'+"Generated "+str(number_of_shadows) + " out of " +
              str(number_of_shadows)+" estimations.", end='')

    output_file.close()
    shadow_file.close()


# given inverse channel eigenvalues in the form of MPS, constructs an MPO representation of the inverse measurement channel

def MinverseMPO(M_inverse):
    size = len(M_inverse)
    bond_dimension = np.shape(M_inverse)[1]
    swap = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    U = (np.array([[1, 0, 0, 1], [0, 1, 1, 0], [
         0, -1j, 1j, 0], [1, 0, 0, -1]])/np.sqrt(2))

    idswap = np.kron(np.identity(2), np.kron(swap, np.identity(2)))
    basis_change = idswap@np.kron(U, U)@idswap
    tensors = []
    for i in range(size):
        local_matrices = np.empty(
            (bond_dimension, bond_dimension, 16, 16), dtype=complex)
        for k in range(bond_dimension):
            for j in range(bond_dimension):
                local_matrices[k][j] = np.conj(np.transpose(
                    basis_change))@np.diag([M_inverse[i][k][j][min(x, 1)] for x in range(16)])@basis_change
        tensors.append(local_matrices)
    return MPO(tensors=tensors)
