import numpy as np

# TENSOR NETWORK IMPLEMENTATION

# MPO CLASS

# an MPO is stored as a list of tensors, the attribute tensors has a shape (size, bond_dimension, bond_dimension, physical_dimension, physical_dimension)
# note that this code only supports MPOs where each tensor has the same bond dimension
# we call self.__tensors[s][i][j][:][:] the matrix in the site s at indices i,j, and self.__tensors[s][:][:][i][j] the bond matrix at site s and indices i,j

# an MPO is initialized by either giving the tensors, as in MPO(tensors= T), where T is of the shape described above,
# or by giving a size, bond dimension and physical dimension, in which case the tensors are filled with zeros.

# MPS are the same, the tensors are given by a list of shape (size, bond_dimension, bond_dimension, physical_dimension),
# self.__tensors[s][i][j][:] is called the vector at site s and indices i,j

class MPO:
    def __init__(self, *, tensors=[], size=None, bond_dimension=None, physical_dimension=None):
        if (len(tensors) == 0):
            if (None in [size, bond_dimension, physical_dimension]):
                raise ValueError(
                    "Size, bond dimension and physical dimension must be specified for random MPO")
            self.__size = size
            self.__bond_dimension = bond_dimension
            self.__physical_dimension = physical_dimension

            self.__tensors = np.zeros((self.__size, self.__bond_dimension, self.__bond_dimension,
                                   self.__physical_dimension, self.__physical_dimension))

        else:
            if ([size, bond_dimension, physical_dimension] != [None, None, None]):
                raise ValueError(
                    "Cannot specify size, bond dimension or physical dimension if tensors are given explicitely")

            self.__tensors = tensors[:]
            self.__size = len(tensors)
            self.__bond_dimension = np.shape(tensors[0])[0]
            self.__physical_dimension = np.shape(tensors[0][0][0])[0]

    def getMatrix(self, site, i, j):
        return self.__tensors[site][i][j]

    def getTensors(self):
        return self.__tensors

    def setMatrix(self, site, i, j, matrix):
        if (np.shape(matrix) != (self.__physical_dimension, self.__physical_dimension)):
            raise ValueError(
                "The shape of this matrix is incompatible with the physical dimension")

        self.__tensors[site][i][j] = matrix

    def setTensors(self, tensors):
        self.__bond_dimension = np.shape(tensors[0])[0]
        self.__physical_dimension = np.shape(tensors[0][0])[0]
        self.__size = len(tensors)
        self.__tensors = tensors

    def getBondDimension(self):
        return self.__bond_dimension

    def getPhysicalDimension(self):
        return self.__physical_dimension

    def getSize(self):
        return self.__size

    # compute the matrix element of the matrix M represented by the MPO, <x|M|y>, where 0≤x,y≤2^size-1
    def matrixElement(self, x, y):
        x_bin = int2list(x, self.__size, self.__physical_dimension)
        y_bin = int2list(y, self.__size, self.__physical_dimension)
        transfer_matrix = np.identity(self.__bond_dimension)
        for i in range(self.__size):
            matrix = np.zeros(
                (self.__bond_dimension, self.__bond_dimension), dtype=complex)
            for a in range(self.__bond_dimension):
                for b in range(self.__bond_dimension):
                    matrix[a, b] = self.__tensors[i][a][b][x_bin[i], y_bin[i]]

            transfer_matrix = transfer_matrix@matrix

        return np.trace(transfer_matrix)

    def bondMatrix(self, site, index1, index2):
        return np.array([np.array([self.__tensors[site][x][y][index1, index2] for x in range(self.__bond_dimension)]) for y in range(self.__bond_dimension)])


class MPS:
    def __init__(self, *, tensors=[], size=None, bond_dimension=None, physical_dimension=None):
        if (len(tensors) == 0):
            if (None in [size, bond_dimension, physical_dimension]):
                raise ValueError(
                    "Size, bond dimension and physical dimension must be specified for random MPO")
            self.__size = size
            self.__bond_dimension = bond_dimension
            self.__physical_dimension = physical_dimension

            self.__tensors = np.zeros((self.__size, self.__bond_dimension,
                                   self.__bond_dimension, self.__physical_dimension), dtype=complex)

        else:
            if ([size, bond_dimension, physical_dimension] != [None, None, None]):
                raise ValueError(
                    "Cannot specify size, bond dimension or physical dimension if tensors are given explicitely")

            self.__tensors = tensors[:]
            self.__size = len(tensors)
            self.__bond_dimension = np.shape(tensors[0])[0]
            self.__physical_dimension = np.shape(tensors[0][0][0])[0]

    def getVector(self, site, i, j):
        return self.__tensors[site][i][j]

    def getTensors(self):
        return self.__tensors

    def setVector(self, site, i, j, vector):
        if (np.shape(vector) != (self.__physical_dimension,)):
            raise ValueError(
                "The shape of this vector is incompatible with the physical dimension")
        self.__tensors[site][i][j] = vector

    def setTensor(self, site, tensor):
        if (np.shape(tensor) != (self.__bond_dimension, self.__bond_dimension, self.__physical_dimension)):
            raise ValueError("Wrong tensor shape")
        self.__tensors[site] = tensor[:]

    def setTensors(self, tensors):
        self.__bond_dimension = np.shape(tensors[0])[0]
        self.__physical_dimension = np.shape(tensors[0][0])[0]
        self.__size = len(tensors)
        self.__tensors = tensors[:]

    def getBondDimension(self):
        return self.__bond_dimension

    def getPhysicalDimension(self):
        return self.__physical_dimension

    def getSize(self):
        return self.__size

    # multiply the MPS by a single site matrix
    def multiplyByMatrix(self, site, matrix):
        for j in range(self.__bond_dimension):
            for k in range(self.__bond_dimension):
                self.__tensors[site][j][k] = matrix@self.__tensors[site][j][k]

    def vectorElement(self, k):  # gets the vector element <k|MPS>, where 0≤k≤2^size-1
        k_bin = int2list(k, self.__size, self.__physical_dimension)
        transfer_matrix = np.identity(self.__bond_dimension)
        for site in range(self.__size):
            transfer_matrix = transfer_matrix@self.bondMatrix(
                site, k_bin[site])
        return np.trace(transfer_matrix)

    def bondMatrix(self, site, index):
        return np.array([np.array([self.__tensors[site][x][y][index] for x in range(self.__bond_dimension)]) for y in range(self.__bond_dimension)])


def int2list(n, N, base=2):  # turns an integer into a list of bits
    return list(map(int, list(np.base_repr(n, base).zfill(N))))


def MPSinnerProduct(x, y):
    transfer_matrix = sum(np.kron(x.bondMatrix(0, k), np.conj(
        y.bondMatrix(0, k))) for k in range(x.getPhysicalDimension()))
    for site in range(1, x.getSize()):
        transfer_matrix = transfer_matrix@sum(np.kron(x.bondMatrix(site, k), np.conj(
            y.bondMatrix(site, k))) for k in range(x.getPhysicalDimension()))

    return np.trace(transfer_matrix)


def MPOtimesMPS(A, x):
    size = A.getSize()
    bond_dimension = A.getBondDimension()*x.getBondDimension()
    physical_dimension = A.getPhysicalDimension()
    product = MPS(size=size, bond_dimension=bond_dimension,
                  physical_dimension=physical_dimension)

    for i in range(size):
        for k1 in range(A.getBondDimension()):
            for j1 in range(x.getBondDimension()):
                for k2 in range(A.getBondDimension()):
                    for j2 in range(x.getBondDimension()):
                        product.setVector(i, k1+A.getBondDimension()*j1, k2+A.getBondDimension(
                        )*j2, A.getMatrix(i, k1, k2)@x.getVector(i, j1, j2))
    return product


def MPOtimesMPO(A, B):
    product = MPO(size=A.getSize(), bond_dimension=A.getBondDimension(
    )*B.getBondDimension(), physical_dimension=A.getPhysicalDimension())

    for i in range(A.getSize()):
        for k1 in range(A.getBondDimension()):
            for j1 in range(B.getBondDimension()):
                for k2 in range(A.getBondDimension()):
                    for j2 in range(B.getBondDimension()):
                        product.setMatrix(i, k1+A.getBondDimension()*j1, k2+A.getBondDimension(
                        )*j2, A.getMatrix(i, k1, k2)@B.getMatrix(i, j1, j2))

    return product


def MPStensorProduct(M1, M2):  # computes |M1> \otimes |M2> as an MPS
    product = MPS(size=M1.getSize(), bond_dimension=M1.getBondDimension(
    )*M2.getBondDimension(), physical_dimension=M1.getPhysicalDimension()*M2.getPhysicalDimension())

    for i in range(M1.getSize()):
        for j1 in range(M1.getBondDimension()):
            for k1 in range(M2.getBondDimension()):
                for j2 in range(M1.getBondDimension()):
                    for k2 in range(M2.getBondDimension()):
                        product.setVector(i, j1+M1.getBondDimension()*k1, j2+M1.getBondDimension(
                        )*k2, np.kron(M1.getVector(i, j1, j2), np.conj(M2.getVector(i, k1, k2))))
    return product


def vectorizeMPO(M):  # given an MPO, returns an MPS corresponding to its vectorization
    vectorized_MPO = MPS(M.getSize(), M.getBondDimension(),
                         M.getPhysicalDimension()**2)
    for i in range(M.getSize()):
        for j in range(M.getBondDimension()):
            for k in range(M.getBondDimension()):
                vectorized_MPO.setVector(i, j, k, [M.getMatrix(i, j, k)[x, y] for x in range(
                    M.getPhysicalDimension()) for y in range(M.getPhysicalDimension())])
    return vectorized_MPO
