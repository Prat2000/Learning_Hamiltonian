import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
import math

J = 1
max_t = 2.05
min_t = 0.05
lattice_shape = (8,8) #It can be  changed to (16,16) or (32,32)
steps = 1
iters_per_step = 31 #to remove autocorrelation among each of the 32 lattice sites
random_state = 50
t_vals = np.linspace(min_t, max_t, 32)
print(t_vals)

# betas = 1 / T_vals
lattices = []
#Monte Carlo Simulation
for beta in t_vals:
        lat=[]
        print(beta)
        random_state=random_state+1
        xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,
                                       beta=1/beta,J=J,random_state=random_state)
        for q in range(40000):
            xy.simulate(steps,iters_per_step)
            lat.append(xy.L+0)
            # draw_grid(lattice_shape[0],xy.L,1/beta)
        lattices.append(lat[30000:])  #initial 30000 rejected and last 10000 accepted
        print('Done')
#Saving Data
output = open('8x8'+'lattices.pkl', 'wb')
pickle.dump(lattices, output)
output.close()