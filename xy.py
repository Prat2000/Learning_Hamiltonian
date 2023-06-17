import numpy as np

class XYModelMetropolisSimulation:
    '''H_matrix is valid only for 2D model'''

    def __init__(self,lattice_shape,beta,J=1,random_state=None):
        self.beta = beta
        self.rs = np.random.RandomState(seed=random_state)
        self.L = self.rs.rand(*lattice_shape)
        self.lattice_shape = lattice_shape
        self.initial_L = self.L.copy()
        self.t = 0
        self.J = J
        self.modified_in_last_step = False

        self.H_matrix = np.zeros(self.L.shape)        
        self._calculate_H_matrix()
        
        
        self.H = np.sum(self.H_matrix) / 2
        self.H_vals = [self.H]
        

    def _calculate_H_matrix(self):  #4x4 neighbourhood
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                self.H_matrix[i, j] = 0
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]])) #periodic boundary ensured through %
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        self.H_matrix *= self.J
        
    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return -ans * self.J
    
    def _renew_H_matrix(self, pos, new_val):
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            link_delta_H = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H_matrix[tuple(pos_list)] -= link_delta_H * self.J
            self.H_matrix[pos] -= link_delta_H * self.J
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            link_delta_H = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H_matrix[tuple(pos_list)] -= link_delta_H * self.J
            self.H_matrix[pos] -= link_delta_H * self.J
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
    
    def make_step(self):
        change_pos = tuple([self.rs.randint(_) for _ in self.lattice_shape]) #which lattice site is changed
        new_val = self.rs.rand() #samples random num between [0,1) uniformly
        delta_H = self._get_delta_H(change_pos, new_val) #change in energy
        if (delta_H > 0):
            if (self.rs.rand() < np.exp(-self.beta * delta_H)): #accepted
                self._renew_H_matrix(change_pos, new_val)
                self.L[change_pos] = new_val
                self.H += delta_H
                self.modified_in_last_step = True
            else:
                self.modified_in_last_step = False
        else: 
            self._renew_H_matrix(change_pos, new_val)
            self.L[change_pos] = new_val
            self.H += delta_H
            self.modified_in_last_step = True
        self.t += 1


    def simulate(self, steps, iters_per_step):
        for i in range(steps):
            for j in range(iters_per_step):
                self.make_step()
            self.H_vals.append(self.H)
            
            
            
   

    