from ._ext import weighted_laplace_sum_forward as _forward


def weighted_laplace_sum(q, k, v):
    """Exact O(N log N) weighted Laplace sum (forward only, fp32, CUDA).

    q: (P, D, M), k: (P, D, N), v: (P, N, C), float32 contiguous, C <= 128, N, M <= 131072.
    Returns (out_w, out_u):
        out_w[p, m, c] = sum_{d, n} exp(-|k[p, d, n] - q[p, d, m]|) * v[p, n, c]    (P, M, C)
        out_u[p, m]    = sum_{d, n} exp(-|k[p, d, n] - q[p, d, m]|)                 (P, M)
    No autograd. Padded keys: zero k and v there and subtract n_pad * sum_d exp(-|q_d|) from out_u.
    """
    return _forward(q, k, v)


__all__ = ["weighted_laplace_sum"]
