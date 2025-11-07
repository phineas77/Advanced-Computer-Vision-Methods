from utils.ex4_utils import *
from utils.ex2_utils import *
import numpy as np
import sympy as sp

from utils.tracker import Tracker



def to_numpy_matrix(matrices):
    return tuple(np.array(matrix).astype(np.float64) for matrix in matrices)


def derive_state_matrices(F, L):
    T, q = sp.symbols('T q')
    Fi = sp.exp(F * T)

    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    return Q, Fi


def get_state_matrices(model_name, dt, q, r):
    R = np.eye(2) * r
    if model_name == 'NCV':
        Fi, H, Q = get_ncv_matrices()
    elif model_name == 'NCA':
        Fi, H, Q = get_nca_matrices()
    elif model_name == 'RW':
        Fi, H, Q = get_rw_matrices()
    else:
        raise ValueError('Unknown model name')

    T, q_sym = sp.symbols('T q')
    Q = Q.subs({T: dt, q_sym: q})
    Fi = Fi.subs({T: dt})
    return to_numpy_matrix((Fi, H, Q, R))


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

    return (Fi, H, Q)


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

    return (Fi, H, Q)


def get_rw_matrices():
    H = np.array([[1, 0],
                  [0, 1]])
    T, q = sp.symbols('T q')
    F = sp.Matrix([[0, 0],
                   [0, 0]])

    L = sp.Matrix([[1, 0],
                   [0, 1]])

    Q, Fi = derive_state_matrices(F, L)
    return Fi, H, Q


def x0y0wh_to_center_wh(region):
    """Convert [x, y, w, h] to (center, (w, h))."""
    x, y, w, h = region
    center = (x + w // 2, y + h // 2)
    return center, (w, h)


def center_wh_to_x0y0wh(coordinates):
    (x_center, y_center), w, h = coordinates

    x0 = x_center - w // 2
    y0 = y_center - h // 2

    return x0, y0, w, h


def hellinger_distance(histograms, q):
    return np.sqrt(np.sum((np.sqrt(histograms) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2)


class ParticleTrackerParams:
    def __init__(self):
        self.dynamic_model = 'NCV'
        self.dt = 1
        self.n_bins = 16
        self.sigma = 1.0
        self.q_cov = 0.1
        self.r_cov = 1
        self.num_particles = 150
        self.alpha = 0.05
        self.hellinger_sigma = 0.15
        self.scale_factor = 0.8


class ParticleTracker(Tracker):
    def __init__(self, params=ParticleTrackerParams()):
        super().__init__()
        self.parameters = params
        if self.parameters.dynamic_model == 'RW':
            self.y_state_index = 1
        elif self.parameters.dynamic_model == 'NCV':
            self.y_state_index = 2
        elif self.parameters.dynamic_model == 'NCA':
            self.y_state_index = 3
        else:
            raise ValueError('Invalid dynamic model')

        self.w = None
        self.h = None
        self.patch_position = None
        self.kernel = None
        self.q = None
        self.A = None
        self.C = None
        self.Q = None
        self.R = None
        self.state = None
        self.particles = None
        self.weights = None
        self.img_shape = None

    def name(self):
        return "Particle-Tracker-N-50-scale-0.8-sigma-0.5-100-NCA"

    def init_state(self):
        if self.parameters.dynamic_model == 'RW':
            return np.array(self.patch_position)
        elif self.parameters.dynamic_model == 'NCV':
            return np.array([self.patch_position[0], 0, self.patch_position[1], 0])
        elif self.parameters.dynamic_model == 'NCA':
            return np.array([self.patch_position[0], 0, 0, self.patch_position[1], 0, 0])
        else:
            raise ValueError('Invalid dynamic model')

    def init_particles(self):
        return sample_gauss(self.state,
                            self.Q,
                            self.parameters.num_particles)

    def remove_particles_outside_image(self):
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_shape[1])
        self.particles[:, self.y_state_index] = np.clip(self.particles[:, self.y_state_index], 0, self.img_shape[0])

    def clip_state(self):
        if self.parameters.dynamic_model == 'RW':
            self.state = np.clip(self.state, 0, [self.img_shape[1], self.img_shape[0]])
        elif self.parameters.dynamic_model == 'NCV':
            self.state[0] = np.clip(self.state[0], 0, self.img_shape[1])
            self.state[2] = np.clip(self.state[2], 0, self.img_shape[0])
        elif self.parameters.dynamic_model == 'NCA':
            self.state[0] = np.clip(self.state[0], 0, self.img_shape[1])
            self.state[3] = np.clip(self.state[3], 0, self.img_shape[0])

    def initialize(self, image, region):
        region = [int(el) for el in region]
        center, (w, h) = x0y0wh_to_center_wh(region)
        w, h = int(w * self.parameters.scale_factor), int(h * self.parameters.scale_factor)
        w = w + 1 if w % 2 == 0 else w
        h = h + 1 if h % 2 == 0 else h

        self.img_shape = image.shape
        self.w = w
        self.h = h
        self.patch_position = list(center)

        patch, _ = get_patch(image, self.patch_position, (w, h))
        self.kernel = create_epanechnik_kernel(w, h, self.parameters.sigma)
        q = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        self.q = q / np.sum(q)

        # self.init_region = init_region
        self.weights = np.ones((self.parameters.num_particles, 1)) / self.parameters.num_particles
        q_cov = self.parameters.q_cov * min(self.w, self.h)

        self.A, self.C, self.Q, self.R = get_state_matrices(self.parameters.dynamic_model,
                                                            self.parameters.dt, q_cov,
                                                            self.parameters.r_cov)

        self.state = self.init_state()
        self.particles = self.init_particles()

    def track(self, image):
        # patch, _ = get_patch(image, self.patch_position, (self.w, self.h))
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.num_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), :]

        noise = sample_gauss(np.zeros(self.state.shape), self.Q, self.parameters.num_particles)
        self.particles = np.matmul(self.particles, self.A.T) + noise
        self.remove_particles_outside_image()

        patches = [get_patch(image, (int(p[0]), int(p[self.y_state_index])), (self.w, self.h))[0]
                   for p in self.particles]
        histograms = [extract_histogram(patch, self.parameters.n_bins, self.kernel) for patch in patches]
        histograms = np.array(histograms)

        histograms = histograms / np.sum(histograms, axis=1, keepdims=True)
        distances = hellinger_distance(histograms, self.q)
        self.weights = np.exp(-0.5 * distances / self.parameters.hellinger_sigma ** 2)
        self.weights = self.weights / np.sum(self.weights)
        self.state = np.sum(self.weights.reshape(-1, 1) * self.particles, axis=0)

        self.clip_state()
        patch, _ = get_patch(image, (int(self.state[0]), int(self.state[self.y_state_index])),
                             (self.w, self.h))
        q_new = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        q_new = q_new / np.sum(q_new)
        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * q_new

        return center_wh_to_x0y0wh(((self.state[0], self.state[self.y_state_index]),
                                    self.w,
                                    self.h))