import torch
import numpy as np
def WAS_xfer(z, wave_len, img_size, pixel_dim):
    n,m = img_size[0:2]
    sx  = wave_len / (pixel_dim * m)
    sy  = wave_len / (pixel_dim * n)
    iy, ix = np.ogrid[0:1+n//2,0:1+m//2]
    x   = (sx * ix)**2
    y   = (sy * iy)**2
    del ix,iy
   # calculate 1 quadrant
    Kj_Z   = z * 2.0j * np.pi / wave_len
    tmp    = np.exp(Kj_Z * np.sqrt(1.0 - x - y))
    tmp_Q = 2.0*np.pi*np.sqrt(1.0 - x - y)
    # and mirror
    tmp    = np.concatenate((tmp[:,:-1],tmp[:,:0:-1]),axis=1)
    tmp_Q    = np.concatenate((tmp_Q[:,:-1],tmp_Q[:,:0:-1]),axis=1)
  # and copy
    Q_out = np.zeros((n,m),dtype='double')
    Q_out[:n//2,:] = tmp_Q[  :-1,:]
    Q_out[n//2:,:] = tmp_Q[:0:-1,:]
    T_out = np.zeros((n,m),dtype='complex128')
    T_out[:n//2,:] = tmp[  :-1,:]
    T_out[n//2:,:] = tmp[:0:-1,:]
    return T_out, Q_out
class Convolve_fast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, T):
        #input in spatial domain, T in Fourier domain
        Xfourier = torch.fft(input,2)
        ctx.save_for_backward(T,Xfourier)
        output = torch.ifft(complex_mul(Xfourier,T.repeat(input.shape[0],1,1,1)),2)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        Tkernel, X_F = ctx.saved_tensors
        gradout_Fourier = torch.fft(grad_output,2)
        grad_input = torch.ifft(complex_mul(gradout_Fourier,complex_cj(Tkernel.repeat(X_F.shape[0],1,1,1))),2)
        grad_T = torch.sum(complex_mul(gradout_Fourier,complex_cj(X_F)),dim=0)/(Tkernel.shape[0]**2)
        return grad_input, grad_T
