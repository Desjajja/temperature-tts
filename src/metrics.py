
import math

def comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def pass_at_k(N, C, K):
    if N == 0 or K == 0:
        return 0.0
    if C == 0:
        return 0.0
    return 1.0 - (comb(N - C, K) / comb(N, K))

def avg_accuracy(N, C):
    return (C / N) if N else 0.0
