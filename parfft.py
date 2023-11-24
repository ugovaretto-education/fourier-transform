import multiprocessing as mp
import ctypes
import numpy as np
import numpy.typing as npt
import numba as nb

@nb.njit(cache=True)
def bitreverse(n: int, num_bits: int) -> int:
    x = 0
    for i in range(num_bits):
        x |= ((n >> i) & 0x1) << ((num_bits - i) - 1)
        #n >>= 1
    return x
@nb.njit(cache=True)
def shuffle_coeff(x: npt.NDArray[np.float64], y: npt.NDArray[np.cdouble]):
    bits = int(np.log2(len(x)))
    for i in range(len(x)):
        y[i] = x[bitreverse(i, bits)] # + 0j
    return y
    
# Numba does not understand any multiprocessing type it is therefore required to accelerate only
# the per-level computation
@nb.njit(cache=True)
def fftlevel(y: npt.NDArray[np.cdouble], level: int, start: int, subrange: int):
    span = 1 << level # span = 2^level
    #if span larger than subrange double size of subrange
    if span > subrange:
        subrange *= 2
    end = start + subrange
    #if offset not a multiple of span do not do any work in this process
    if start % span != 0:
        end = 0 #this causes the loop to terminate immediately and jump directly to the barrier.wait statement
    s2 = span >> 1 # half span, iterate until span/2 and use Eq. 1
    wm = np.exp(-1j * np.pi / s2) # s2 == span/2 --> np.exp(-2j*np.pi/span) = np.exp(-j*np.pi/(span/2)) = np.exp(-j*np.pi/s2) 
    w = 1. + 0j
    #print(f"{mp.current_process().name} level: {l}, offset: {start}, span: {span}, subrange: {subrange}")
    for s in range(start, start + s2): # iterate from offset to offset + span / 2 - 1
        for k in range(s, end, span):
            #w = np.exp(-2j*np.pi*(s + i * span)/span) #k is a multiple of span, therefore, it can be set outside the loop                                                #see Eq. 2
            u = y[k]
            y[k] = u + w * y[k + s2]
            y[k + s2] = u - w * y[k + s2]
        w *= wm

# Numba does not understand any multiprocessing type, it is therefore required to accelerate only
# the per-level computation
def accelfft(x: mp.RawArray, fft_range: tuple[int, int], barrier: mp.Barrier) -> None: # -> npt.NDArray[np.cdouble]:
    y = np.frombuffer(x, dtype=np.cdouble)
    N = len(y)
    levels = int(np.log2(N))
    start = fft_range[0]
    subrange = fft_range[1]
    for l in range(1, levels + 1):
        fftlevel(y, l, start, subrange)
        barrier.wait()
        #print(f"{mp.current_process().name}, y[0]: {y[0]}")

# apply fft to subrange
# y contains the already shuffled elements
def subfft(x: mp.RawArray, fft_range: tuple[int, int], barrier: mp.Barrier) -> None: # -> npt.NDArray[np.cdouble]:
    y = np.frombuffer(x, dtype=np.cdouble)
    N = len(y)
    levels = int(np.log2(N))
    start = fft_range[0]
    subrange = fft_range[1]
    for l in range(1, levels + 1):
        span = 1 << l # span = 2^level
        #if span larger than subrange double size of subrange
        if span > subrange:
            subrange *= 2
        end = start + subrange
        #if offset not a multiple of span do not do any work in this process
        if start % span != 0:
            end = 0 #this causes the loop to terminate immediately and jump directly to the barrier.wait statement
        s2 = span >> 1 # half span, iterate until span/2 and use Eq. 1
        wm = np.exp(-1j * np.pi / s2) # s2 == span/2 --> np.exp(-2j*np.pi/span) = np.exp(-j*np.pi/(span/2)) = np.exp(-j*np.pi/s2) 
        w = 1. + 0j
        #print(f"{mp.current_process().name} level: {l}, offset: {start}, span: {span}, subrange: {subrange}")
        for s in range(start, start + s2): # iterate from offset to offset + span / 2 - 1
            for k in range(s, end, span):
                #w = np.exp(-2j*np.pi*(s + i * span)/span) #k is a multiple of span, therefore, it can be set outside the loop                                                #see Eq. 2
                u = y[k]
                y[k] = u + w * y[k + s2]
                y[k + s2] = u - w * y[k + s2]
            w *= wm
        barrier.wait()
        #print(f"{mp.current_process().name}, y[0]: {y[0]}")

#if jit == True use Numba accelerated code
def parfft(x: npt.NDArray[np.float64], num_tasks: int, jit: bool = False) -> npt.NDArray[np.cdouble]:
    N = len(x)
    assert (np.log2(N) - np.floor(np.log2(N))) == 0
    assert len(x) % num_tasks == 0
    y = np.zeros(N, dtype = np.cdouble)
    shuffle_coeff(x, y)
    z = mp.RawArray(ctypes.c_double, 2 * len(y))
    z_np = np.frombuffer(z, dtype=np.cdouble)
    np.copyto(z_np, y)
    subrange = N // num_tasks
    barrier = mp.Barrier(num_tasks)
    processes = [mp.Process(target=accelfft if jit else subfft, args=(z, (i*subrange, subrange), barrier)) for i in range(num_tasks)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    y = np.frombuffer(z, dtype=np.cdouble)
    return y

#if jit == True use Numba accelerated code
def inplace_parfft(x: npt.NDArray[np.float64], num_tasks: int, z: mp.RawArray, jit: bool = False) -> None:
    N = len(x)
    assert N == len(z) // 2
    assert (np.log2(N) - np.floor(np.log2(N))) == 0
    assert len(x) % num_tasks == 0
    z_np = np.frombuffer(z, dtype=np.cdouble)
    shuffle_coeff(x, z_np)
    subrange = N // num_tasks
    barrier = mp.Barrier(num_tasks)
    processes = [mp.Process(target=accelfft if jit else subfft, args=(z, (i*subrange, subrange), barrier)) for i in range(num_tasks)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        
