"""Randomised degradation pipeline used to synthesise (corrupted, clean) training pairs.

All functions operate on float32 RGB images in [0, 1] with shape (H, W, 3) and take a
numpy Generator so that every DataLoader worker draws independent random streams.

The pipeline covers every degradation family listed in the brief (Gaussian, Poisson,
salt-and-pepper and speckle noise; Gaussian, motion and defocus blur; JPEG artifacts;
chromatic aberration; mixtures) plus several *extra* families (Rician / MRI noise,
uniform noise, row/column banding, spatially-correlated noise, resolution loss,
quantisation) so the model learns a general "restore" prior instead of memorising
a fixed list of inverse operators.
"""
import math

import cv2
import numpy as np

cv2.setNumThreads(0)  # DataLoader workers provide the parallelism


# --------------------------------------------------------------------------- blur kernels
def _norm(k):
    return (k / k.sum()).astype(np.float32)


def gaussian_kernel(rng, max_sigma=3.5):
    """Isotropic or anisotropic (rotated) Gaussian kernel."""
    sx = rng.uniform(0.3, max_sigma)
    sy = sx if rng.random() < 0.5 else rng.uniform(0.3, max_sigma)
    theta = rng.uniform(0, math.pi)
    size = int(2 * math.ceil(3 * max(sx, sy)) + 1)
    size = max(3, min(size, 31))
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = c * x + s * y, -s * x + c * y
    return _norm(np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)))


def motion_kernel(rng, max_len=25):
    """Linear or slightly curved motion-blur kernel at a random angle."""
    length = int(rng.integers(3, max_len + 1))
    size = length + 2 + (length + 1) % 2
    k = np.zeros((size, size), np.float32)
    c = size // 2
    angle = rng.uniform(0, math.pi)
    curve = rng.uniform(-0.3, 0.3) if rng.random() < 0.3 else 0.0
    n = length * 4
    for i in range(n):
        t = (i / (n - 1) - 0.5) * length
        a = angle + curve * t / max(length, 1)
        x, y = c + t * math.cos(a), c + t * math.sin(a)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            k[yi, xi] += 1.0
    if k.sum() == 0:
        k[c, c] = 1
    return _norm(k)


def defocus_kernel(rng, max_radius=6.0):
    """Disk (out-of-focus) kernel, anti-aliased."""
    r = rng.uniform(1.0, max_radius)
    size = int(2 * math.ceil(r) + 1)
    c = size // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    d = np.sqrt(x ** 2 + y ** 2)
    return _norm(np.clip(r + 0.5 - d, 0, 1))


def apply_blur(img, rng):
    kind = rng.choice(["gaussian", "motion", "defocus"], p=[0.4, 0.3, 0.3])
    if kind == "gaussian":
        k = gaussian_kernel(rng)
    elif kind == "motion":
        k = motion_kernel(rng)
    else:
        k = defocus_kernel(rng)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- noise
def gaussian_noise(img, rng):
    sigma = rng.uniform(1, 55) / 255.0
    if rng.random() < 0.35:   # luminance-only noise
        n = rng.standard_normal(img.shape[:2], dtype=np.float32)[..., None]
    else:
        n = rng.standard_normal(img.shape, dtype=np.float32)
    return img + sigma * n


def poisson_noise(img, rng):
    peak = 10 ** rng.uniform(0.7, 2.7)   # ~5 .. 500 photons at white
    if rng.random() < 0.35:
        gray = img.mean(2, keepdims=True)
        noisy = rng.poisson(np.clip(gray, 0, 1) * peak).astype(np.float32) / peak
        return img + (noisy - gray)
    return rng.poisson(np.clip(img, 0, 1) * peak).astype(np.float32) / peak


def speckle_noise(img, rng):
    std = rng.uniform(0.03, 0.4)
    if rng.random() < 0.5:
        n = rng.standard_normal(img.shape[:2], dtype=np.float32)[..., None]
    else:
        n = rng.standard_normal(img.shape, dtype=np.float32)
    return img + img * n * std


def salt_pepper_noise(img, rng):
    amount = rng.uniform(0.002, 0.1)
    salt_ratio = rng.uniform(0.2, 0.8)
    out = img.copy()
    shape = img.shape[:2] if rng.random() < 0.6 else img.shape  # whole pixel vs per channel
    m = rng.random(shape)
    salt = m < amount * salt_ratio
    pepper = (m >= amount * salt_ratio) & (m < amount)
    out[salt] = 1.0
    out[pepper] = 0.0
    return out


def rician_noise(img, rng):
    """MRI magnitude noise."""
    sigma = rng.uniform(2, 30) / 255.0
    n1 = rng.standard_normal(img.shape, dtype=np.float32) * sigma
    n2 = rng.standard_normal(img.shape, dtype=np.float32) * sigma
    return np.sqrt((img + n1) ** 2 + n2 ** 2)


def uniform_noise(img, rng):
    a = rng.uniform(2, 40) / 255.0
    return img + rng.uniform(-a, a, img.shape).astype(np.float32)


def correlated_noise(img, rng):
    """Spatially correlated (demosaicing-like / film-grain-like) noise."""
    sigma = rng.uniform(3, 35) / 255.0
    n = rng.standard_normal(img.shape, dtype=np.float32)
    n = cv2.GaussianBlur(n, (0, 0), rng.uniform(0.5, 1.5))
    n /= n.std() + 1e-8
    return img + sigma * n


def banding_noise(img, rng):
    """Row / column fixed-pattern noise (sensor read-out stripes) plus light white noise."""
    h, w = img.shape[:2]
    s = rng.uniform(2, 20) / 255.0
    if rng.random() < 0.5:
        band = rng.standard_normal((h, 1, 1)).astype(np.float32)
    else:
        band = rng.standard_normal((1, w, 1)).astype(np.float32)
    return img + s * band + rng.uniform(0, 8) / 255.0 * rng.standard_normal(img.shape, dtype=np.float32)


NOISES = {
    "gaussian": (gaussian_noise, 0.26),
    "poisson": (poisson_noise, 0.18),
    "speckle": (speckle_noise, 0.12),
    "salt_pepper": (salt_pepper_noise, 0.12),
    "rician": (rician_noise, 0.08),
    "uniform": (uniform_noise, 0.06),
    "correlated": (correlated_noise, 0.10),
    "banding": (banding_noise, 0.08),
}
_NOISE_NAMES = list(NOISES)
_NOISE_P = np.array([NOISES[k][1] for k in _NOISE_NAMES])
_NOISE_P /= _NOISE_P.sum()


def apply_noise(img, rng):
    name = rng.choice(_NOISE_NAMES, p=_NOISE_P)
    return NOISES[name][0](img, rng)


# --------------------------------------------------------------------------- other artifacts
def jpeg(img, rng):
    q = int(rng.integers(5, 96))
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def chromatic_aberration(img, rng):
    """Lateral CA: R and B channels radially scaled / shifted relative to G."""
    h, w = img.shape[:2]
    out = img.copy()
    cx, cy = w / 2 + rng.uniform(-0.1, 0.1) * w, h / 2 + rng.uniform(-0.1, 0.1) * h
    for ch, sign in ((0, 1.0), (2, -1.0)):
        scale = 1.0 + sign * rng.uniform(0.002, 0.02)
        tx, ty = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
        m = np.array([[scale, 0, (1 - scale) * cx + tx], [0, scale, (1 - scale) * cy + ty]], np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3:   # longitudinal CA: blur one colour channel a little
        ch = int(rng.integers(0, 3))
        out[..., ch] = cv2.GaussianBlur(out[..., ch], (0, 0), rng.uniform(0.5, 1.5))
    return out


def resolution_loss(img, rng):
    h, w = img.shape[:2]
    s = rng.uniform(1.3, 4.0)
    down = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
    up = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
    small = cv2.resize(img, (max(8, int(w / s)), max(8, int(h / s))), interpolation=int(down))
    return cv2.resize(small, (w, h), interpolation=int(up))


def quantization(img, rng):
    levels = 2 ** int(rng.integers(3, 7))
    return np.round(np.clip(img, 0, 1) * (levels - 1)) / (levels - 1)


# --------------------------------------------------------------------------- pipeline
def degrade(img, rng, strength=1.0):
    """Apply a random chain of degradations.

    Order loosely follows image formation: optics (blur, CA) -> sampling -> sensor
    noise -> compression -> impulse noise. Returns a float32 image quantised to the
    uint8 grid, exactly like the competition's uint8 inputs.
    """
    if rng.random() < 0.03:          # occasional clean input: learn identity / no over-smoothing
        return quantize_u8(img)

    x = img
    applied = 0
    if rng.random() < 0.45 * strength:
        x = apply_blur(x, rng); applied += 1
    if rng.random() < 0.2 * strength:
        x = chromatic_aberration(x, rng); applied += 1
    if rng.random() < 0.12 * strength:
        x = resolution_loss(x, rng); applied += 1
    if rng.random() < 0.8 or applied == 0:
        x = apply_noise(x, rng); applied += 1
        if rng.random() < 0.2 * strength:   # mixed noise
            x = apply_noise(x, rng)
    x = np.clip(x, 0, 1)
    if rng.random() < 0.05:
        x = quantization(x, rng)
    if rng.random() < 0.35 * strength:
        x = jpeg(x, rng)
    if rng.random() < 0.06 * strength:
        x = salt_pepper_noise(x, rng)
    return quantize_u8(x)


def quantize_u8(x):
    return np.round(np.clip(x, 0, 1) * 255.0).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- content augmentation
def augment_content(img, rng, lowlight_p=0.15, gray_p=0.08):
    """Geometric + photometric augmentation of the *clean* target.

    Low-light simulation darkens the clean image itself (gamma + gain) so the model
    sees dark content at evaluation time; the corruption is then applied on top,
    which yields the high relative noise typical of low-light photos.
    """
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
    if rng.random() < gray_p:
        img = np.repeat(img.mean(2, keepdims=True), 3, axis=2)
    if rng.random() < lowlight_p:
        img = np.power(img, rng.uniform(1.0, 2.2)) * rng.uniform(0.08, 0.6)
    return np.ascontiguousarray(img, dtype=np.float32)
