
import h5py
import numpy as np
import pdb  # Debugger import
import cv2
import matplotlib.pyplot as plt


def get_data(problem, **kwargs):
    
    
    match problem:


        case "sPGD_test":

            # sPGD data test:

            mat_fname = "offline_tt.mat"   
            f = h5py.File(mat_fname,'r')
            x_train = np.array(f['res_train'])
            x_train = np.transpose(x_train)

            x_test = np.array(f['res_test'])
            x_test = np.transpose(x_test)

            p_train = np.array(f['param_train'])
            p_train = np.transpose(p_train)

            p_test = np.array(f['param_test'])
            p_test = np.transpose(p_test)


            
            return x_train, x_test, p_train, p_test


        case "Surface_2D":


            npz_fname = "Surface_2D.npz"
            f = np.load(npz_fname)
            
            x_train = f['x_train']   # [Dofs x Samples]
            x_test  = f['x_test']    # [Dofs x Samples]
            p_train = f['p_train']   # [Parameters x Samples]
            p_test  = f['p_test']    # [Parameters x Samples]

            return x_train, x_test, p_train, p_test


        
