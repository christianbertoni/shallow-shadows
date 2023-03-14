import numpy as np 
from numpy import array, kron, trace, transpose, identity, zeros, shape, reshape, empty
from numpy.random import random
from scipy.linalg import pinv, lstsq
import h5py
import sys
from tensor_networks import MPS


##### CODE FOR INVERTING A THE MEASUREMENT CHANNEL

#inputs:
#size: size of the input MPS, this corresponds to half of the number of qubits.
#depth: depth of the circuit
#target_accuracy: distance between approximate inverse and actual inverse required
#output_file_name: name of the file where the inverse will be saved

#M_ansatz: ansatz for the inverse, if empty, a random ansatz is chosen.
#regularize: if True, the regularization guiding the optimization towards a translationally invariant inverse is applied
#verbose: if True, the cost function is displayed at every step
#brute_force_cost: if True, the cost function is computed inefficiently by summing all of the 2^n terms (m(x)v(x)-1)^2 

def constructInverse(size, depth, bond_dimension, target_accuracy, output_file_name, M_ansatz=[], regularize=True, verbose=True, brute_force_cost=False):
	if(size==1):
		regularize=False

	M=measurementMap(size,depth)

	if(len(M_ansatz)==0):
		M_ansatz=MPS(tensors=[random((bond_dimension,bond_dimension,2)) for k in range(size)])

	M_inverse=invert(M,M_ansatz,target_accuracy, regularize, verbose, brute_force_cost)

	output_file=h5py.File(output_file_name, 'w')
	output_file.create_dataset('inverse_MPS', shape=(size,bond_dimension, bond_dimension,2), data=M_inverse.getTensors())
	output_file.close()



def invert(M,M_ansatz, target_accuracy, regularize, verbose, brute_force_cost):

	M_inverse=M_ansatz
	cost=costFunction(M,M_inverse,0,brute_force_cost)
	
	while(cost>target_accuracy):
		for site in range(M.getSize()):

			new_tensor=empty((2,M_inverse.getBondDimension(),M_inverse.getBondDimension()))
			for index in range(2):
				dist=costFunction(M,M_inverse,0,brute_force_cost)
				regularization_strength=int(regularize)*min(dist,1)
				cost=costFunction(M,M_inverse,regularization_strength,brute_force_cost)
				
				new_tensor[index]=optimizeDirection(M,M_inverse,site,index, regularization_strength)

			M_inverse.setTensor(site, transpose([x for x in new_tensor]))
			

			if(verbose):
				print('\r'+"                                           ", end='')
				print('\r'+"Distance from inverse: "+str(costFunction(M,M_inverse,0,brute_force_cost)), end='')
				
                
	if(verbose):
				print('\r'+"Distance from inverse: "+str(costFunction(M,M_inverse,0,brute_force_cost)), end='')	
	return M_inverse

####### MEASUREMENT MAP COMPUTATION

##this function computes the MPS representation of the measurement channel M, i.e. the MPS that we need to invert

def measurementMap(size, depth):
    T_s=transpose(array([[1,0,0,0],[0, 3/15,3/15,3/15],[0, 3/15,3/15,3/15],[0, 9/15,9/15,9/15]]))

    F=array(transpose([1,1/3,1/3,1/9]))

    T=T_s

    for k in range(1,depth):
        T=kron(T,identity(2))@kron(identity(2**(k)),T_s)

    T=T@transpose(kron(identity(2**(depth-1)),F))

    local_matrices=kron([1,0,0,0], identity(2**(depth-1)))@T, kron([0,1,0,0],identity(2**(depth-1)))@T

    return MPS(tensors=[transpose(local_matrices) for k in range(size)])



####### COST FUNCTION COMPUTATION

### we can compute the cost function in a "brute force" way, simply adding up all 2^n terms (m(x)v(x)-1)^2, this is useful for debugging and to benchmark the
### floating point error problems discussed in the paper

def bruteForceCost(M,M_inverse):
	size=M.getSize()
	cost=0
	clist=[]
	for k in range(2**size):
		bit_string_k=list(map(int,list(np.base_repr(k,2).zfill(size))))
		m=M.bondMatrix(0,bit_string_k[0])
		m_inverse=M_inverse.bondMatrix(0,bit_string_k[0])
		for j in range(1,size):
			m=m@M.bondMatrix(j,bit_string_k[j])
			m_inverse=m_inverse@M_inverse.bondMatrix(j,bit_string_k[j])
		cost+=(trace(m)*trace(m_inverse)-1)**2
		clist.append((trace(m)*trace(m_inverse)-1)**2)
	return sum(x**2 for x in clist)

### efficient computation of the cost function.

def costFunction(M, M_inverse, regularization_strength=0, brute_force=False):
	if(brute_force):
		return bruteForceCost(M,M_inverse)
	size=M.getSize()
	linear_part=np.identity(M.getBondDimension()*M_inverse.getBondDimension())
	quadratic_part=np.identity(len(linear_part)**2)
	for site in range(size):
		quadratic_part=quadratic_part@sum(kron(kronSquare(M.bondMatrix(site,index)), kronSquare(M_inverse.bondMatrix(site,index))) for index in range(2))   
		linear_part=linear_part@sum(kron(M.bondMatrix(site,index), M_inverse.bondMatrix(site,index)) for index in range(2))
	
	return trace(quadratic_part)-2*trace(linear_part)+2**size+regularization_strength*translationalVariance(M_inverse)

### additional cost function measuring how far the MPS is from translational invariance, used for regularization 

def translationalVariance(M_inverse):
	average_tensors=[zeros((M_inverse.getBondDimension(),M_inverse.getBondDimension())),zeros((M_inverse.getBondDimension(),M_inverse.getBondDimension())) ]
	for index in range(2):
		for site in range(M_inverse.getSize()):
			average_tensors[index]+=M_inverse.bondMatrix(site,index)/M_inverse.getSize()
	r=0
	for site in range(M_inverse.getSize()):
		for index in range(2):
			V=M_inverse.bondMatrix(site,index)-average_tensors[index]
			r+=trace(V@transpose(V))
	return r


####### OPTIMIZATION

### to optimize along one tensor given by a site and an index in {0,1}, we minimize the quadratic form given by |X> -> <X|A|X> + <X|B> + 2^n, i.e. we solve the linear system 2A|X>+|B>=0 

def optimizeDirection(M,M_inverse, site, index, regularization_strength):
	[A,B]=quadraticForm(M,M_inverse,site,index, regularization_strength)

	solution=lstsq(2*A,-B)[0]

	return np.reshape(solution,(M_inverse.getBondDimension(), M_inverse.getBondDimension()))




### compute A=quadratic_part and |B>=linear_part for the quadratic form corresponding to the cost function when we fixed all tensor except V_index^site.

def quadraticForm(M, M_inverse, site, index, regularization_strength=0):
	size=M.getSize()
	bond_dimension=M_inverse.getBondDimension()
	quadratic_transfer_matrix=np.identity((M.getBondDimension()*M_inverse.getBondDimension())**2)
	linear_transfer_matrix=np.identity(M.getBondDimension()*M_inverse.getBondDimension())
	quadratic_part=zeros((bond_dimension**2, bond_dimension**2))
	linear_part=zeros(bond_dimension**2)

	for k in range(site):
		quadratic_transfer_matrix=quadratic_transfer_matrix@sum(kron(kronSquare(M.bondMatrix(k,s)), kronSquare(M_inverse.bondMatrix(k,s))) for s in range(2))   
		linear_transfer_matrix=linear_transfer_matrix@sum(kron(M.bondMatrix(k,s), M_inverse.bondMatrix(k,s)) for s in range(2))
	for k in range(size-1,site,-1):
		quadratic_transfer_matrix=sum(kron(kronSquare(M.bondMatrix(k,s)), kronSquare(M_inverse.bondMatrix(k,s))) for s in range(2))@quadratic_transfer_matrix 
		linear_transfer_matrix=sum(kron(M.bondMatrix(k,s), M_inverse.bondMatrix(k,s)) for s in range(2))@linear_transfer_matrix


	for i in range(bond_dimension):
		for j in range(bond_dimension):
			for k in range(bond_dimension):
				for l in range(bond_dimension):
					quadratic_part[j+bond_dimension*i,l+bond_dimension*k]= trace(quadratic_transfer_matrix@kron(kronSquare(M.bondMatrix(site,index)), kron(matrixBasis(i,j,bond_dimension),matrixBasis(k,l,bond_dimension))))

	for i in range(bond_dimension):
		for j in range(bond_dimension):
			linear_part[j+bond_dimension*i]=-2*trace(linear_transfer_matrix@kron(M.bondMatrix(site,index),matrixBasis(i,j,bond_dimension)))

	##regularization part

	if(size>1):
		reg_matrix=regularization_strength*identity(bond_dimension**2)*(1-1/size)**2
		reg_vector=-regularization_strength*reshape(sum(M_inverse.bondMatrix(s,index) for s in range(size) if s!=site), bond_dimension**2)*2*(1-1/size)*1/size

		return [quadratic_part+reg_matrix,linear_part+reg_vector]
	else: 
		return [quadratic_part,linear_part]


### other useful functions



def kronSquare(matrix):
	return kron(matrix,np.conj(matrix))


def matrixBasis(i,j,n):
    m=zeros((n,n))
    m[i,j]=1.
    return m



