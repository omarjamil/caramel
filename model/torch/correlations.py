from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import data_io
import normalize

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
            "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch/",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

region = "50S69W"
 # Data normalizer class
nt = normalize.Normalizers(locations)
# Training and testing data class
nn_data = data_io.Data_IO(region, locations)           

def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = signal.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = signal.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return corr, max_corr, argmax_corr - shift

def timeseriers_correlation(tseries1, tseries2):
    """
    """
    corr = signal.correlate(tseries1, tseries2, mode='same')
    # corr = signal.convolve(tseries1, tseries2, mode='same')

    return corr

def smooth(x,window_len=6,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y
   
def nooverlap_smooth(arrayin, window=6):
    """
    Moving average with non-overlapping window
    """
    x,y=arrayin.shape
    print(x,y)
    averaged = np.mean(arrayin.reshape(window,x//window,y),axis=0)
    return averaged

if __name__ == "__main__":
    start,end = 0,3600
    q_ = nn_data.q_norm_test[start:end,:]
    q_raw =  nn_data.q_test_raw[start:end,:] 
    qphys = nn_data.qphys_dot_norm_test[start:end,:]
    qadv = nn_data.qadv_norm_test[start:end,:]
    qadv_dot = nn_data.qadv_dot_norm_test[start:end,:]
    qadv_raw = nn_data.qadv_test_raw[start:end,:] # qadv_normaliser.inverse_transform(qadv)
    qadv_dot_raw = nn_data.qadv_dot_test_raw[start:end,:]
    # qadv_inv = nt.inverse_minmax(qadv, qadv_scale.data.numpy(), qadv_feature_min.data.numpy(), qadv_feature_max.data.numpy(), qadv_data_min.data.numpy()) 
    qadv_inv = nt.inverse_std(qadv, nt.qadv_stdscale.data.numpy(), nt.qadv_mean.data.numpy())
    # qadv_inv = nn_data.qadv_test_raw[start:end,:]
    qphys_inv = nt.inverse_std(qphys, nt.qphys_dot_stdscale.data.numpy(), nt.qphys_dot_mean.data.numpy())

    # qadv_dot_inv = qadv_dot_raw * 600.
    qadv_dot_inv = qadv_dot_raw *600.
    print(qadv_dot_raw.shape, qadv_dot_inv.shape)
    t1 = qadv_dot_inv[:]
    t2 = qphys_inv[:]
    # corr = timeseriers_correlation(t1,t2)
    # xcorr, maxxcor, lag = cross_corr(t1, t2)
    # print(xcorr, maxxcor, lag)
    fig, (ax1, ax2, ax_corr) = plt.subplots(3, 1, sharex=False)
    t1_s = nooverlap_smooth(t1)
    print(np.mean(t1), np.mean(t1_s))
    ax1.plot(t1[:,2])
    ax2.plot(t1_s[:,2])
    ax_corr.plot(np.sum(t1,axis=0))
    ax_corr.plot(np.sum(t1_s,axis=0))
    fig.tight_layout()
    plt.show()