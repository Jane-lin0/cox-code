import numpy as np

''' Global constants and defaults'''
QUIET = 0
MAX_ITER = 1000
ABSTOL = 1e-4
RELTOL = 1e-2


''' Data preprocessing'''
m, n = A.shape

# save a matrix-vector multiply
Atb = A.T.dot(b)

# check that sum(p) = total number of elements in x
if sum(p) != n:
    raise ValueError('invalid partition')

# cumulative partition
cum_part = np.cumsum(p)


'''ADMM solver '''
x = np.zeros(n)
z = np.zeros(n)
u = np.zeros(n)

# pre-factor
L, U = factor(A, rho)  # You need to define the 'factor' function accordingly

if not QUIET:
    print('%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

for k in range(1, MAX_ITER + 1):

    # x-update
    q = Atb + rho * (z - u)  # temporary value
    if m >= n:  # if skinny
        x = np.linalg.solve(U, np.linalg.solve(L, q))
    else:  # if fat
        x = q / rho - np.linalg.solve(A.T @ np.linalg.solve(U, np.linalg.solve(L, A @ q)), rho ** 2)

    # z-update
    zold = z.copy()
    start_ind = 0
    x_hat = alpha * x + (1 - alpha) * zold
    for i in range(len(p)):
        sel = slice(start_ind, cum_part[i])
        z[sel] = shrinkage(x_hat[sel] + u[sel], lambda_ / rho)  # You need to define the 'shrinkage' function accordingly
        start_ind = cum_part[i]
    u += (x_hat - z)

    # diagnostics, reporting, termination checks
    obj_val = objective(A, b, lambda_, cum_part, x, z)  # You need to define the 'objective' function accordingly

    r_norm = np.linalg.norm(x - z)
    s_norm = np.linalg.norm(-rho * (z - zold))

    eps_pri = np.sqrt(n) * ABSTOL + RELTOL * max(np.linalg.norm(x), np.linalg.norm(-z))
    eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u)

    if not QUIET:
        print('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' % (k, r_norm, eps_pri, s_norm, eps_dual, obj_val))

    if r_norm < eps_pri and s_norm < eps_dual:
        break

if not QUIET:
    toc(t_start)  # You need to define 'toc' function accordingly


def objective(A, b, lambda_, cum_part, x, z):
    obj = 0
    start_ind = 0
    for i in range(len(cum_part)):
        sel = slice(start_ind, cum_part[i])
        obj += np.linalg.norm(z[sel])
        start_ind = cum_part[i]
    p = (1/2 * np.sum((A @ x - b)**2) + lambda_ * obj)
    return p


def shrinkage(x, kappa):
    return np.maximum(0, 1 - kappa / np.linalg.norm(x)) * x


def factor(A, rho):
    m, n = A.shape
    if m >= n:  # if skinny
        L = np.linalg.cholesky(A.T @ A + rho * np.eye(n)).T
    else:  # if fat
        L = np.linalg.cholesky(np.eye(m) + 1/rho * (A @ A.T))
    # force python to recognize the upper / lower triangular structure
    L = np.matrix(L)
    U = L.T
    return L, U
