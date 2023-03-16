# shallow-shadows

This is the code implementing the shallow shadows protocol as detailed in [arXiv:2209.12924](https://arxiv.org/abs/2209.12924). 

Prerequisites:  
-numpy  
-scipy  
-h5py  
-qiskit 

All results are saved in h5py files. Refer to the Jupyter notebook <code>example.ipynb</code> for a practical example.

## Tensor network logic

The protocol is based on Matrix Product States and Operators. The file <code>tensor_networks.py</code> defines classes for MPS and MPO and implements basic operations one can do with these objects. 

## Data acquisition

The file <code>data_acquisition.py</code> contains code to simulate the process of data acquisition for MPS input states. Given a state, a circuit depth and a number of snapshots, it generates a classical shadow.

## Expectation estimation

The file <code>expectation_estimation.py</code> contains code to use the shadow generated in the data acquisition step to estimate expectation values of observables. You can estimate either shallow or sparse observable, as detailed in the paper. The former requires to specify the vectorization of the observable as an MPS, the latter requires a list of Pauli operators and coefficients.

## Inverse MPS

To estimate shallow observables, one needs to generate an MPS representation of the inverse measurement channel. The file <code>inverse_MPS</code> contains code to do that. This procedure only needs to be done once for each size and depth, hence we provide pre computed inverses for a number of cases in <code>data/inverses</code>.
