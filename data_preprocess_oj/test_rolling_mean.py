import numpy as np
import matplotlib.pyplot as plt

def array_diff(inarray: np.array, step: int, axis: int=0):
    """
    Calculate element wise difference for 
    the array with a given step. Therefore
    the elements for difference do not have to consecutive
    """
    # Create a slice same shape as the in array
    slc1 = [slice(None)]*len(inarray.shape)
    slc2 = [slice(None)]*len(inarray.shape)
    end=-1*step
    axis_length = np.size(inarray,axis=axis)
    # The required axis will an actual slice over it
    # the other dimensions will be None which is the same as slice(:)
    slc1[axis] = slice(0,end)
    slc2[axis] = slice(step,axis_length)
    # Create two arrays and then take the difference
    a1 = inarray[tuple(slc1)]
    a2 = inarray[tuple(slc2)]
    diff = a2 - a1
    return diff

def nooverlap_smooth(arrayin, window=10):
    """
    Moving average with non-overlapping window
    """
    x,y=arrayin.shape
    averaged = np.mean(arrayin.reshape(window,x//window,y),axis=0)
    return averaged

def reconstruct(x,y,z, nsteps):
    """
    reconstruct time seriese
    """
    
    x_ = np.zeros((nsteps,5))
    x_[0,:] = x[0,:]
    for i in range(1,nsteps):
        x_[i,:] = x_[i-1] + y[i-1,:] + z[i-1,:]
    return x_

def contruct_tseries(x,y,z,nsteps):
    """
    """
    x_ = np.zeros((nsteps,5))
    x_[0,:] = x[0,:]
    for i in range(nsteps):
        x_[i,:] = x_[i-1,:] + y[i-1,:] + z[i-1,:]

    return x_

if __name__ == "__main__":
    nsteps=100000
    x = np.random.uniform(-1,1,(nsteps,5))
    y = np.random.uniform(-1,1,(nsteps,5))
    z = np.random.uniform(-1,1,(nsteps,5))
    a =  contruct_tseries(x,y,z, nsteps)
    y_ave = nooverlap_smooth(y)
    z_ave = nooverlap_smooth(z)
    a_ave = nooverlap_smooth(a)
    print(a_ave.shape)
    nsteps_,_=a_ave.shape
    a_ave_ = reconstruct(a_ave,y_ave,z_ave, nsteps_)
    print(a_ave_.shape)
    plt.plot(a_ave[:,1],'r-.')
    plt.plot(a_ave_[:,1], 'k-.')
    plt.show()
