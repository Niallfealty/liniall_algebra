""" 'Naive' algorithm (borrowed to understand)
"""
import numpy as np
from scipy.linalg import hessenberg
from tabulate import tabulate
from tqdm import tqdm

try:
    from utils import monitoring_report
except ImportError:
    print("Failed to import utils")

def gram_schmidt_qr(A, Q):
    ''' Gram-Schmidt for computing QR decomposition
    '''
    R = np.eye(*A.shape)
    for i in range(A.shape[0]):
        v_i = A[i].copy()
        for j in range(i-1):
            R[j,i] = Q[j].conj() @ A[i]
            v_i = v_i - R[j,i]*Q[j]
        R[i,i] = v_i @ v_i.T
        Q[i] = v_i/R[i,i]

    return Q, R

def qr_factorisation(Ak_prime, Qk, smult):
    Q, R = np.linalg.qr(Ak_prime)
    Ak_plus = np.add(R @ Q, smult)
    QQ_plus = Qk @ Q
    return Ak_plus, QQ_plus

def qr_step(Ak, Qk, n, k, monitor):
    ''' Single QR fact iteration '''
    # pick the subtraction factor to be the last element on the diagonal
    s = Ak.item(n-1, n-1)
    smult = s * np.eye(n)
    Ak_prime = np.subtract(Ak, smult) # use Ak_prime for quicker convergence

    Ak_plus, QQ_plus = qr_factorisation(Ak_prime, Qk, smult)

    # "peek" into the structure of matrix A from time to time
    monitoring_report(Ak_plus, k, 10_000, monitor)

    return Ak_plus, QQ_plus

def eigen_qr(A, iterations=500000, monitor=False):
    # get A into Hessenberg form to aid convergence
    # TODO: write hessenberg function
    Ak, Q = hessenberg(A, calc_q=True)

    n = Ak.shape[0]
    QQ = np.eye(n) # initial QQ
    iters = range(iterations) if monitor else tqdm(range(iterations))
    for k in iters:
        Ak, QQ = qr_step(Ak, QQ, n, k, monitor)
    return Ak, QQ

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Experimenting and running unit tests on QR decomposition tools")
    parser.add_argument("N", type=int, help="dimension of NXN matrix (n.b. randomly chosen so no guarantee that this equals rank)")
    parser.add_argument("-i", "--iterations", type=int, default=50000, help="Number of QR iterations, default is 50,000")
    parser.add_argument("-m", "--monitor", action="store_true", help="Output progress on every 1k iterations")
    parser.add_argument("-s", "--symmetric", action="store_true", help="Test with a symmetric matrix")
    args = parser.parse_args()

    # A is a square random matrix of size n

    A = np.random.rand(args.N, args.N)
    if args.symmetric:
        A = A + A.transpose()
    print("A=")
    print(tabulate(A))
    Ak, QQ = eigen_qr(A, args.iterations, args.monitor)
    print(np.diag(Ak))
    print(np.linalg.eig(A)[0])

    for qr_comp, np_comp in zip(sorted(np.diag(Ak)), sorted(np.linalg.eig(A)[0])):
        print(f"==> computed: {qr_comp:.9}\tactual: {np_comp:.9}")

if __name__ == "__main__": main()
