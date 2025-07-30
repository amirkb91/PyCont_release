import numpy as np
import scipy.linalg as spl


def bifurcation_functions(M):
    # floquet multipliers
    floq = np.sort(spl.eigvals(M))
    stability = 1.0 if np.all(np.abs(floq) <= 1) else 0.0

    # Bifurcation test functions
    phi_fold = np.sign(np.linalg.det(M - np.eye(len(M))))
    phi_flip = np.sign(np.linalg.det(M + np.eye(len(M))))

    n = len(floq)
    phi_NS = 1
    for i in range(n):
        for j in range(i + 1, n):
            phi_NS *= floq[i] * floq[j] - 1
    phi_NS = np.sign(np.real(phi_NS))

    # floq_conjugates_index = np.where(np.diff(np.real(floq)) == 0)[0]
    # phi_NS = np.prod(floq[floq_conjugates_index] * floq[floq_conjugates_index + 1] - 1)
    # phi_NS = np.real(phi_NS)

    # Alternative test functions (bialternate product is very expensive)
    # phi_fold = np.real(np.prod(floq - 1))
    # phi_flip = np.real(np.prod(floq + 1))
    # M_bi_M = bialternate_same(M)
    # phi_NS = spl.det(M_bi_M - np.eye(len(M_bi_M)))

    return floq, stability, phi_fold, phi_flip, phi_NS


def get_unstable_eigenvec(M):
    # return the eigenvector corresponding to the single unstable eigenvalue outside the unit circle
    eigvals, eigvecs = spl.eig(M)
    eigvals = np.abs(eigvals)
    idx = np.argmax(eigvals)
    return eigvecs[:, idx]


def bialternate_same(A):
    n = A.shape[0]
    m = int(0.5 * n * (n - 1))
    Bialt = np.zeros((m, m))

    i = 0
    for p in range(1, n):
        for q in range(p):
            j = 0
            for r in range(1, n):
                for s in range(r):
                    Bialt[i, j] = A[p, r] * A[q, s] - A[p, s] * A[q, r]
                    j += 1
            i += 1

    return Bialt


def bialternate(A, B):
    n = A.shape[0]
    m = int(0.5 * n * (n - 1))
    Bialt = np.zeros((m, m))

    i = 0
    for p in range(1, n):
        for q in range(p):
            j = 0
            for r in range(1, n):
                for s in range(r):
                    Bialt[i, j] = 0.5 * (
                        A[p, r] * B[q, s]
                        - A[p, s] * B[q, r]
                        + B[p, r] * A[q, s]
                        - B[p, s] * A[q, r]
                    )
                    j += 1
            i += 1

    return Bialt
