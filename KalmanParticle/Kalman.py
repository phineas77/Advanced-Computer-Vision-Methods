import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sympy as sp
from sympy.interactive.printing import init_printing
from sympy import latex
from sympy import simplify
import pytest

from ex4_utils import *

#Constant velocity model
T = sp.symbols('T')
F = sp.Matrix([[0, 1],[0, 0]])
Fi = sp.exp(F*T)
print(Fi)
print("-------------------------------------------------")
#Nearly constant velocity model
#1D
T, q = sp.symbols('T q')
Fi = sp.Matrix([[1, T],[0, 1]])
L = sp.Matrix([[0], [1]])
Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
print(Q)

#2D
T, q = sp.symbols('T q')
F = sp.Matrix([[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0]])
Fi = sp.exp(F*T)
print(Fi)

print(latex(F))
# print(latex(Fi))

L = sp.Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])
Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
print(Q)

# print(latex(Q))
# print(latex(L))

T, q = sp.symbols('T q')
F = sp.Matrix([[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0]])
Fi = sp.exp(F*T)
print(Fi)
print("-------------------------------------------------")


#Random walk

T, q = sp.symbols('T q')
F = sp.Matrix([[0, 0],[0, 0]])
Fi = sp.exp(F*T)
print(Fi)

L = sp.Matrix([[1, 0], [0, 1]])
Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
print(Q)

# print(latex(Q))

L = sp.Matrix([[1, 0], [0, 0], [0, 1], [0, 0]])
print(L)
print("-------------------------------------------------")

#Nearly constant acceleration
T, q = sp.symbols('T q')
F = sp.Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
Fi = sp.exp(F*T)
print(Fi)

# print(latex(F))
# print(latex(Fi))

L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 1]])
print(L)

Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
print(Q)

# print(latex(Q))
# print(latex(L))

T, q = sp.symbols('T q')
F = sp.Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
Fi = sp.exp(F*T)
print(Fi)

L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
print(L)

Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
Q_simplified = simplify(Q)
print(Q_simplified)
print("-------------------------------------------------")


def to_numpy_matrix(matrices):
    return tuple(np.array(m).astype(np.double) for m in matrices)

def derive_state_matrices(F, L):
    # use the same symbol names T and q
    T, q = sp.symbols('T q')
    Fi = sp.exp(F * T)
    # factor q outside the integral
    Q = q * sp.integrate(Fi * L * (Fi * L).T, (T, 0, T))
    return Q, Fi

def get_ncv_matrices():
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    F = sp.Matrix([[0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    L = sp.Matrix([[0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 1]])
    Q, Fi = derive_state_matrices(F, L)
    return Fi, H, Q

def get_nca_matrices():
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])
    F = sp.Matrix([[0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0]])
    L = sp.Matrix([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0],
                   [0, 1]])
    Q, Fi = derive_state_matrices(F, L)
    return Fi, H, Q

def get_rw_matrices():
    H = np.array([[1, 0],
                  [0, 1]])
    F = sp.zeros(2)
    L = sp.eye(2)
    Q, Fi = derive_state_matrices(F, L)
    return Fi, H, Q

def get_state_matrices(model_name, dt, q, r):
    # same symbol names for substitution
    T, q_sym = sp.symbols('T q')
    R = np.eye(2) * r

    if model_name == 'NCV':
        Fi, H, Q = get_ncv_matrices()
    elif model_name == 'NCA':
        Fi, H, Q = get_nca_matrices()
    elif model_name == 'RW':
        Fi, H, Q = get_rw_matrices()
    else:
        raise ValueError(f"Unknown model name '{model_name}'")

    # substitute numerical dt and q
    Fi = Fi.subs(T, dt)
    Q  = Q.subs({T: dt, q_sym: q})

    return to_numpy_matrix((Fi, H, Q, R))


print(get_rw_matrices()[0])

A, C, Q_i, R_i = get_state_matrices('NCV', 1, 1, 1)
print(A, C, Q_i, R_i)

N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
sx[0] = x[0]
sy[0] = y[0]
state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
state[0] = x[0]
state[1] = y[0]
covariance = np.eye(A.shape[0], dtype=np.float32)
for j in range(1, x.size):
    state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                          np.reshape(state, (-1, 1)), covariance)
    sx[j] = state[0, 0]
    sy[j] = state[1, 0]

plt.plot(x,y, color='blue', label='True trajectory', marker='o')
plt.plot(sx, sy, color='red', label='Estimated trajectory', marker='o')
plt.show()





A, C, Q_i, R_i = get_state_matrices('NCA', 1, 2, 1)
N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
sx[0] = x[0]
sy[0] = y[0]
state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
state[0] = x[0]
state[1] = y[0]
covariance = np.eye(A.shape[0], dtype=np.float32)
for j in range(1, x.size):
    state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                          np.reshape(state, (-1, 1)), covariance)
    # print(state)
    sx[j] = state[0, 0]
    sy[j] = state[3, 0]

def run_kalman(model_type, trajectory, q, r):
    x_vals, y_vals = trajectory
    A, C, Q_i, R_i = get_state_matrices(model_type, 1, q, r)
    N = len(x_vals)

    # prepare output
    est_x = np.empty(N, dtype=np.float32)
    est_y = np.empty(N, dtype=np.float32)

    # pick which state‐index holds y
    idx_map = {'RW': 1, 'NCV': 2, 'NCA': 3}
    y_idx   = idx_map.get(model_type)
    if y_idx is None:
        raise ValueError(f"Unknown model type '{model_type}'")

    # init
    est_x[0] = x_vals[0]
    est_y[0] = y_vals[0]

    dim   = A.shape[0]
    state = np.zeros((dim, 1), dtype=np.float32)
    state[0, 0]     = x_vals[0]
    state[y_idx, 0] = y_vals[0]

    P = np.eye(dim, dtype=np.float32)

    for j in range(1, N):
        # stack new measurement into a (2×1) column
        z = np.vstack((x_vals[j], y_vals[j])).astype(np.float32)

        state, P, _, _ = kalman_step(
            A, C, Q_i, R_i,
            z, state, P
        )

        # extract scalars directly
        est_x[j] = state[0, 0]
        est_y[j] = state[y_idx, 0]

    return est_x, est_y


# for model_type in ['RW', 'NCV', 'NCA']:
#     for r, q in [(1, 1), (1, 10), (1, 100), (10, 1), (100, 5)]:
#         sx, sy = run_kalman(model_type, (x, y), q, r)
#         plt.plot(sx, sy, label=f'{model_type} q={q} r={r}', marker='o', color='red')
#         plt.plot(x,y, color='blue', label='True trajectory', marker='o')
#         plt.show()
#     sx, sy = run_kalman(model_type, (x, y), 1, 1)
#     plt.plot(sx, sy, label=model_type)

def test_kalman(A, C, Q_i, R_i, N=40):
    # generate a shrinking spiral
    t = np.linspace(5 * np.pi, 0, N)
    true_x = t * np.cos(t)
    true_y = t * np.sin(t)

    # prepare output arrays
    est_x = np.empty(N, dtype=np.float32)
    est_y = np.empty(N, dtype=np.float32)
    est_x[0], est_y[0] = true_x[0], true_y[0]

    # initialize state (column vector) and covariance
    dim = A.shape[0]
    state = np.zeros((dim, 1), dtype=np.float32)
    state[0, 0], state[1, 0] = true_x[0], true_y[0]
    P = np.eye(dim, dtype=np.float32)

    # run through measurements
    for k, (mx, my) in enumerate(zip(true_x[1:], true_y[1:]), start=1):
        z = np.array([[mx], [my]], dtype=np.float32)
        state, P, _, _ = kalman_step(A, C, Q_i, R_i, z, state, P)
        est_x[k], est_y[k] = state[0, 0], state[1, 0]

    return true_x, true_y, est_x, est_y


model_types = ['RW', 'NCV', 'NCA']
combinations = [(1, 1), (1, 10), (1, 100), (10, 1), (100, 5)]
num_rows = len(model_types)
num_cols = len(combinations)

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Loop through each combination and plot
for i, model_type in enumerate(model_types):
    for j, (r, q) in enumerate(combinations):
        sx, sy = run_kalman(model_type, (x, y), q, r)
        axs[i, j].plot(sx, sy, label=f'{model_type} q={q} r={r}', marker='o', color='red', alpha=0.5)
        axs[i, j].plot(x, y, color='blue', label='True trajectory', marker='o', alpha=0.5)
        # axs[i, j].legend()
        axs[i, j].set_title(f'{model_type} q={q} r={r}')

plt.tight_layout()
#plt.savefig('./plots/kalman_spiral.pdf')
plt.show()


N = 50
v = np.linspace(0, 8 * np.pi, N)
x = v
y = np.mod(v, 2*np.pi) + np.mod(v, 1.5*np.pi) + np.mod(v, 0.8*np.pi)

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Jagged Path')
plt.grid(True)
plt.show()


model_types = ['RW', 'NCV', 'NCA']
combinations = [(1, 1), (1, 10), (1, 100), (10, 1), (100, 5)]
num_rows = len(model_types)
num_cols = len(combinations)

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Loop through each combination and plot
for i, model_type in enumerate(model_types):
    for j, (r, q) in enumerate(combinations):
        sx, sy = run_kalman(model_type, (x, y), q, r)
        axs[i, j].plot(sx, sy, label=f'{model_type} q={q} r={r}', marker='o', color='red', alpha=0.5)
        axs[i, j].plot(x, y, color='blue', label='True trajectory', marker='o', alpha=0.5)
        # axs[i, j].legend()
        axs[i, j].set_title(f'{model_type} q={q} r={r}')

plt.tight_layout()
#plt.savefig('./plots/kalman_jagged.pdf')
plt.show()


x_corners = [0, 5, 5, 0, 0]
y_corners = [0, 0, 3, 3, 0]
x = np.array(x_corners)
y = np.array(y_corners)
plt.plot(x_corners, y_corners, 'r-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rectangle Path')
plt.grid(True)
plt.axis('equal')
plt.show()


model_types = ['RW', 'NCV', 'NCA']
combinations = [(1, 1), (1, 10), (1, 100), (10, 1), (100, 5)]
num_rows = len(model_types)
num_cols = len(combinations)

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Loop through each combination and plot
for i, model_type in enumerate(model_types):
    for j, (r, q) in enumerate(combinations):
        sx, sy = run_kalman(model_type, (x, y), q, r)
        axs[i, j].plot(sx, sy, label=f'{model_type} q={q} r={r}', marker='o', color='red', alpha=0.5)
        axs[i, j].plot(x, y, color='blue', label='True trajectory', marker='o', alpha=0.5)
        # axs[i, j].legend()
        axs[i, j].set_title(f'{model_type} q={q} r={r}')

plt.tight_layout()
#plt.savefig('./plots/kalman_rectangle.pdf')
plt.show()
