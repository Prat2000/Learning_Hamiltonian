import pickle as pkl
import numpy as np

file = open('8x8lattices.pkl','rb')
lattice_set = pkl.load(file)

t = 0.05 + 5*(2.0/32)
print(t)
#10k samples each of shape (8,8)
lattices = lattice_set[5]

#flattened lattice shape (64,) => eql input
lat_flat = np.ndarray.flatten(lattices[0])
print(lat_flat.shape)
print(lat_flat[:10])