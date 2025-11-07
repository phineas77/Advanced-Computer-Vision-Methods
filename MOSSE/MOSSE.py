import numpy as np
import cv2
from utils.tracker import Tracker
from utils.ex2_utils import get_patch
from utils.ex3_utils import create_cosine_window, create_gauss_peak


def x0y0wh_to_center_wh(region):
    """Convert [x, y, w, h] to (center, (w, h))."""
    x, y, w, h = region
    center = (x + w // 2, y + h // 2)
    return center, (w, h)

def center_wh_to_x0y0wh(center, size):
    """Convert center and (w, h) to [x, y, w, h]."""
    cx, cy = center
    w, h = size
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    return [x, y, w, h]

class MOSSETracker(Tracker):
    def __init__(self, sigma=5, lambda_=0.1, alpha=0.2, scaling_factor=1.5):
        super().__init__()
        # Parameters for correlation filter and update
        self.sigma = sigma         # Gaussian peak parameter
        self.lambda_ = lambda_     # Regularization constant
        self.alpha = alpha         # Update speed (exponential forgetting)
        self.scaling_factor = scaling_factor  # Factor to enlarge patch for filter computation

        # Variables to be set during initialization
        self.original_size = None  # (w, h) of the target (forced odd)
        self.scaled_size = None    # (w, h) after applying scaling_factor (forced odd)
        self.patch_position = None # Current target center (as a list, for updating)
        self.cos_window = None     # Cosine window for the scaled patch
        self.G = None              # FFT of the desired Gaussian response
        self.H_conj = None         # Correlation filter in Fourier domain

    def name(self):
        return "MOSSETracker_SF_{:.2f}".format(self.scaling_factor)

    def preprocess_patch(self, patch):
        """
        Preprocess the image patch:
         - Convert to grayscale,
         - Apply logarithm,
         - Normalize to zero mean and unit variance.
        """
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        patch = np.log1p(patch)
        mean_val = np.mean(patch)
        std_val = np.std(patch) + 1e-5
        return (patch - mean_val) / std_val

    def _force_odd(self, size):
        """Force both dimensions in size to be odd."""
        w, h = size
        if w % 2 == 0:
            w += 1
        if h % 2 == 0:
            h += 1
        return (w, h)

    def _scale_size(self, size):
        """Apply the scaling factor and force the dimensions to be odd."""
        w, h = size
        new_w = int(w * self.scaling_factor)
        new_h = int(h * self.scaling_factor)
        return self._force_odd((new_w, new_h))

    def initialize(self, img, region: list):
        """
        Initialize the tracker using the first frame and initial bounding box.
        :param img: Initial frame (BGR image)
        :param region: Bounding box [x, y, w, h] (top-left coordinates)
        """
        # Convert region to integers and to center-based representation
        region = list(map(int, region))
        center, (w, h) = x0y0wh_to_center_wh(region)
        # Force original size to be odd
        self.original_size = self._force_odd((w, h))
        # Save the patch position as a list (to allow updating)
        self.patch_position = list(center)
        # Compute scaled size using the scaling factor
        self.scaled_size = self._scale_size(self.original_size)

        # Extract the patch using the scaled size
        patch, _ = get_patch(img, center, self.scaled_size)
        if (patch.shape[1], patch.shape[0]) != self.scaled_size:
            patch = cv2.resize(patch, self.scaled_size)
        patch = self.preprocess_patch(patch)

        # Create cosine (Hanning) window for the scaled patch
        self.cos_window = create_cosine_window(self.scaled_size)
        patch_windowed = patch * self.cos_window

        # Compute FFT of the windowed patch
        P = np.fft.fft2(patch_windowed)

        # Create the desired Gaussian response and compute its FFT
        gauss = create_gauss_peak(self.scaled_size, self.sigma)
        self.G = np.fft.fft2(gauss)

        # Compute the filter (conjugate form) as in equation (1):
        # Ĥ† = (Ĝ * conj(P̂)) / (P̂ * conj(P̂) + λ)
        self.H_conj = (self.G * np.conj(P)) / (P * np.conj(P) + self.lambda_)

    def track(self, img):
        """
        Track the object in a new frame, update the filter, and return the new bounding box.
        :param img: New frame (BGR image)
        :return: Updated bounding box [x, y, w, h] in original target size.
        """
        # Extract patch from current frame at the current patch position
        patch, _ = get_patch(img, self.patch_position, self.scaled_size)
        if (patch.shape[1], patch.shape[0]) != self.scaled_size:
            patch = cv2.resize(patch, self.scaled_size)
        patch = self.preprocess_patch(patch)
        patch_windowed = patch * self.cos_window

        # Compute FFT of the current patch
        F = np.fft.fft2(patch_windowed)

        # Compute correlation response (localization) using inverse FFT
        response = np.fft.ifft2(self.H_conj * F)
        response = np.real(response)

        # Find the location of the maximum response (displacement)
        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        if dx > self.scaled_size[0] // 2:
            dx -= self.scaled_size[0]
        if dy > self.scaled_size[1] // 2:
            dy -= self.scaled_size[1]

        # Update patch position (center of scaled patch)
        self.patch_position[0] += dx
        self.patch_position[1] += dy

        # Extract new patch and update filter
        patch, _ = get_patch(img, self.patch_position, self.scaled_size)
        if (patch.shape[1], patch.shape[0]) != self.scaled_size:
            patch = cv2.resize(patch, self.scaled_size)
        patch = self.preprocess_patch(patch)
        F = np.fft.fft2(patch * self.cos_window)
        # Compute new filter update from current patch
        H_conj_new = (np.conj(F) * self.G) / (F * np.conj(F) + self.lambda_)
        # Update filter with exponential forgetting (equation (3))
        self.H_conj = (1 - self.alpha) * self.H_conj + self.alpha * H_conj_new

        # Return bounding box in original size (unscaled) using current patch position as center
        return center_wh_to_x0y0wh(self.patch_position, self.original_size)

