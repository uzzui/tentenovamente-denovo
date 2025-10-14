import cv2
import numpy as np
from scipy.spatial import Delaunay
from datetime import datetime
import os

IMG_PATH = "free-palestine-2.jpg"
OUTPUT_FOLDER = "img"
NUM_X = 4 * 10
NUM_Y = 5 * 10
TOTAL_POINTS = 700
EDGE_POINTS_RATIO = 0.5
SATURATION_FACTOR = 1.3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if SATURATION_FACTOR != 1.0:
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:, :, 1] *= SATURATION_FACTOR
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

h, w, _ = img_rgb.shape

gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(gx, gy)

grad_norm = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX)
grad_norm = cv2.GaussianBlur(grad_norm, (9, 9), 0)
grad_norm = cv2.dilate(grad_norm, np.ones((3, 3), np.uint8), iterations=1)

ys, xs = np.mgrid[0:h, 0:w]
coords = np.vstack([xs.ravel(), ys.ravel()]).T

probs = grad_norm.ravel() + 1e-6
probs /= probs.sum()

n_edge_points = int(TOTAL_POINTS * EDGE_POINTS_RATIO)
edge_indices = np.random.choice(len(coords), size=n_edge_points, replace=False, p=probs)
edge_points = coords[edge_indices]

xs_grid = np.linspace(0, w - 1, NUM_X)
ys_grid = np.linspace(0, h - 1, NUM_Y)
grid_x, grid_y = np.meshgrid(xs_grid, ys_grid)
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

points = np.vstack([grid_points, edge_points])

tri = Delaunay(points)
output = np.zeros_like(img_rgb)

for simplex in tri.simplices:
    pts = points[simplex].astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    mean_color = cv2.mean(img_rgb, mask=mask)[:3]
    cv2.fillConvexPoly(output, pts, tuple(map(int, mean_color)))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{OUTPUT_FOLDER}/lowpoly_adaptative_{timestamp}.jpg"
cv2.imwrite(filename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print(f"✅ Saved as: {filename}")
