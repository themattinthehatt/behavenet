# Cython implementation of message passing
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
import torch
import numpy as np
import numpy.random as npr
import scipy.misc as scpm

cimport numpy as np
from cython cimport float
from libc.math cimport log, exp, fmax, INFINITY

cdef float logsumexp(float[::1] x):
    cdef int i, N
    cdef float m, out

    N = x.shape[0]

    # find the max
    m = -INFINITY
    for i in range(N):
        m = fmax(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += exp(x[i] - m)

    return m + log(out)


cdef dlse(float[::1] a,
          float[::1] out):

    cdef int K, k
    K = a.shape[0]
    cdef float lse = logsumexp(a)

    for k in range(K):
        out[k] = exp(a[k] - lse)


# def dLSE_da(a, B):
#     return np.exp(a + B.T - logsumexp(a + B.T, axis=1, keepdims=True))
#
# def vjp_LSE_B(a, B, v):
#     return v * dLSE_da(a, B).T

cpdef forward_pass(float[::1] log_pi0,
                   float[:,:,::1] log_As,
                   float[:,::1] log_likes,
                   float[:,::1] alphas):

    cdef int T, K, t, k
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_As.shape[0] == T-1
    assert log_As.shape[1] == K
    assert log_As.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    cdef float[::1] tmp = np.zeros(K,dtype=np.float32)

    for k in range(K):
        alphas[0, k] = log_pi0[k] + log_likes[0, k]

    for t in range(T - 1):
        for k in range(K):
            for j in range(K):
                tmp[j] = alphas[t, j] + log_As[t, j, k]
            alphas[t+1, k] = logsumexp(tmp) + log_likes[t+1, k]

    return logsumexp(alphas[T-1])


cpdef backward_pass(float[:,:,::1] log_Ps,
                    float[:,::1] log_likes,
                    float[:,::1] betas):

    cdef int T, K, t, k, j
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_Ps.shape[0] == T-1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    cdef float[::1] tmp = np.zeros(K,dtype=np.float32)

    # Initialize the last output
    for k in range(K):
        betas[T-1, k] = 0

    for t in range(T-2,-1,-1):
        # betal[t] = logsumexp(Al + betal[t+1] + aBl[t+1],axis=1)
        for k in range(K):
            for j in range(K):
                tmp[j] = log_Ps[t, k, j] + betas[t+1, j] + log_likes[t+1, j]
            betas[t, k] = logsumexp(tmp)

        
cpdef backward_sample(float[:,:,::1] log_Ps,
                      float[:,::1] log_likes,
                      float[:,::1] alphas,
                      float[::1] us,
                      int[::1] zs):

    cdef int T, K, t, k, j
    cdef float Z, acc

    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_Ps.shape[0] == T-1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K
    assert us.shape[0] == T
    assert zs.shape[0] == T

    cdef float[::1] lpzp1 = np.zeros((K,),dtype=np.float32)
    cdef float[::1] lpz = np.zeros((K,),dtype=np.float32)

    
    for t in range(T-1,-1,-1):
        # compute normalized log p(z[t] = k | z[t+1])
        for k in range(K):
            lpz[k] = lpzp1[k] + alphas[t, k]
        Z = logsumexp(lpz)

        # sample
        acc = 0
        zs[t] = K-1
        for k in range(K):
            acc += np.exp(lpz[k] - Z)
            if us[t] < acc:
                zs[t] = k
                break

        # set the transition potential
        if t > 0:
            for k in range(K):
                lpzp1[k] = log_Ps[t-1, k, zs[t]]


cpdef grad_hmm_normalizer(float[:,:,::1] log_As,
                          float[:,::1] alphas,
                          float[::1] d_log_pi0,
                          float[:,:,::1] d_log_As,
                          float[:,::1] d_log_likes):

    cdef int T, K, t, k, j

    T = alphas.shape[0]
    K = alphas.shape[1]
    assert log_As.shape[0] == d_log_As.shape[0] == T-1
    assert log_As.shape[1] == d_log_As.shape[1] == K
    assert log_As.shape[2] == d_log_As.shape[2] == K
    assert d_log_pi0.shape[0] == K
    assert d_log_likes.shape[0] == T
    assert d_log_likes.shape[1] == K

    # Initialize temp storage for gradients
    cdef float[::1] tmp1 = np.zeros((K,),dtype=np.float32)
    cdef float[:, ::1] tmp2 = np.zeros((K, K),dtype=np.float32)

    dlse(alphas[T-1], d_log_likes[T-1])
    for t in range(T-1, 0, -1):
        # tmp2 = dLSE_da(alphas[t-1], log_As[t-1])
        #      = np.exp(alphas[t-1] + log_As[t-1].T - logsumexp(alphas[t-1] + log_As[t-1].T, axis=1))
        #      = [dlse(alphas[t-1] + log_As[t-1, :, k]) for k in range(K)]
        for k in range(K):
            for j in range(K):
                tmp1[j] = alphas[t-1, j] + log_As[t-1, j, k]
            dlse(tmp1, tmp2[k])


        # d_log_As[t-1] = vjp_LSE_B(alphas[t-1], log_As[t-1], d_log_likes[t])
        #               = d_log_likes[t] * dLSE_da(alphas[t-1], log_As[t-1]).T
        #               = d_log_likes[t] * tmp2.T
        #
        # d_log_As[t-1, j, k] = d_log_likes[t, k] * tmp2.T[j, k]
        #                     = d_log_likes[t, k] * tmp2[k, j]
        for j in range(K):
            for k in range(K):
                d_log_As[t-1, j, k] = d_log_likes[t, k] * tmp2[k, j]

        # d_log_likes[t-1] = d_log_likes[t].dot(dLSE_da(alphas[t-1], log_As[t-1]))
        #                  = d_log_likes[t].dot(tmp2)
        for k in range(K):
            d_log_likes[t-1, k] = 0
            for j in range(K):
                d_log_likes[t-1, k] += d_log_likes[t, j] * tmp2[j, k]

    # d_log_pi0 = d_log_likes[0]
    for k in range(K):
        d_log_pi0[k] = d_log_likes[0, k]


# Python interface
def hmm_sample(log_pi0, log_Ps, ll):

    to_numpy = lambda arr: arr.detach().numpy() if not arr.is_cuda else arr.detach().cpu().numpy()
    log_pi0 = to_numpy(log_pi0)
    log_Ps = to_numpy(log_Ps)
    ll = to_numpy(ll)

    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K),dtype=np.float32)
    forward_pass(log_pi0, log_Ps, ll, alphas)

    # Sample backward
    us = npr.rand(T).astype(np.float32)
    zs = -1 * np.ones(T, dtype=np.int32)
    backward_sample(log_Ps, ll, alphas, us, zs)
    
    return torch.LongTensor(zs)


def hmm_expectations(log_pi0, log_Ps, ll, device):
    T, K = ll.shape

    # Make sure everything is C contiguous
    to_numpy = lambda arr: arr.detach().numpy() if not arr.is_cuda else arr.detach().cpu().numpy()
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
    log_pi0 = to_c(to_numpy(log_pi0))
    log_Ps = to_c(to_numpy(log_Ps))
    ll = to_c(to_numpy(ll))

    alphas = np.zeros((T, K),dtype=np.float32)
    forward_pass(log_pi0, log_Ps, ll, alphas)
    normalizer = scpm.logsumexp(alphas[T-1])

    betas = np.zeros((T, K),dtype=np.float32)
    backward_pass(log_Ps, ll, betas)    

    expected_states = alphas + betas
    expected_states -= scpm.logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)
    
    expected_joints = alphas[:T-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = np.exp(expected_joints)
    expected_joints /= expected_joints.sum((1,2))[:,None,None]
    
    return torch.FloatTensor(expected_states).to(device), \
           torch.FloatTensor(expected_joints).to(device), \
           torch.FloatTensor(np.array(normalizer)).to(device)
