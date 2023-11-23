import argparse as ap
from time import perf_counter
import numpy as np

if __name__ == "__main__":
    parser = ap.ArgumentParser(
                    prog='fft',
                    description='FFT',
                    epilog='Compute the Fast Fourier Transform')
    parser.add_argument('power_of_two', type=int)
    args = parser.parse_args()
    N = 2 ** args.power_of_two
    x = np.arange(1, N+1)
    start = perf_counter()
    _ = np.fft.fft(x)
    stop = perf_counter()
    print(f"{stop-start:.4f}")



