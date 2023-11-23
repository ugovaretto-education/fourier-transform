import numpy as np
import numpy.typing as npt
import numba as nb
import argparse as ap
from time import perf_counter

@nb.njit(cache=True)
def bitreverse(n: int, num_bits: int) -> int:
    x = 0
    for i in range(num_bits):
        x |= ((n >> i) & 0x1) << ((num_bits - i) - 1)
        #n >>= 1
    return x

@nb.njit(cache=True)
def shuffle_coeff(x: npt.NDArray[np.float64], y: npt.NDArray[np.cdouble]) -> None:
    bits = int(np.log2(len(x)))
    for i in range(len(x)):
        y[i] = x[bitreverse(i, bits)] # + 0j

@nb.njit(cache=True)
def fft(x: npt.NDArray[np.float64]) -> npt.NDArray[np.cdouble]:
    N = len(x)
    y = np.zeros(N, dtype = np.cdouble)
    shuffle_coeff(x, y)
    levels = int(np.log2(N))
    for l in range(1, levels + 1):
        span = 1 << l # span = 2^level
        s2 = span >> 1 # half span iterate until span/2 and use Eq. 1
        wm = np.exp(-1j * np.pi / s2) # s2 == span/2 --> np.exp(-2j*np.pi/span) = np.exp(-j*np.pi/(span/2)) = np.exp(-j*np.pi/s2) 
        w = 1. + 0j
        for s in range(s2): # iterate from 0 to span / 2 - 1
            for k in range(s, N, span):
                #w = np.exp(-2j*np.pi*(s + i * span)/span) #k is a multiple of span, therefore, it can be set outside the loop
                                                           #see Eq. 2
                u = y[k]
                y[k] = u + w * y[k + span // 2]
                y[k + span // 2] = u - w * y[k + span // 2]
            w *= wm
    return y  
    
@nb.njit(cache=True)
def inplace_fft(x: npt.NDArray[np.float64], y: npt.NDArray[np.cdouble]) -> None:
    N = len(x)
    shuffle_coeff(x, y)
    levels = int(np.log2(N))
    for l in range(1, levels + 1):
        span = 1 << l # span = 2^level
        s2 = span >> 1 # half span iterate until span/2 and use Eq. 1
        wm = np.exp(-1j * np.pi / s2) # s2 == span/2 --> np.exp(-2j*np.pi/span) = np.exp(-j*np.pi/(span/2)) = np.exp(-j*np.pi/s2) 
        w = 1. + 0j
        for s in range(s2): # iterate from 0 to span / 2 - 1
            for k in range(s, N, span):
                #w = np.exp(-2j*np.pi*(s + i * span)/span) #k is a multiple of span, therefore, it can be set outside the loop
                                                           #see Eq. 2
                u = y[k]
                y[k] = u + w * y[k + span // 2]
                y[k + span // 2] = u - w * y[k + span // 2]
            w *= wm


if __name__ == "__main__":
    parser = ap.ArgumentParser(
                    prog='fft',
                    description='FFT',
                    epilog='Compute the Fast Fourier Transform')
    parser.add_argument('power_of_two', type=int)
    parser.add_argument('repetitions', type=int)
    args = parser.parse_args()
    N = 2 ** args.power_of_two
    reps = args.repetitions
    x = np.arange(1, N+1)
    y = np.empty(N, dtype=np.cdouble)
    # pre-compile
    inplace_fft(x,y)
    start = perf_counter()
    for _ in range(reps):
        inplace_fft(x,y)
    stop = perf_counter()
    assert np.allclose(y, np.fft.fft(x), atol=1e-7, rtol=1e-2)
    print(f"{(stop-start)/reps:.4f} s")
