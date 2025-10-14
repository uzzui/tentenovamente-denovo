import cv2
import numpy as np
from scipy.spatial import Delaunay
from datetime import datetime
import os

FRAC = 0.44  # % points in the grid
NUM_X = 4 * 25
NUM_Y = 5 * 25
IMG_PATH = "free-palestine-3.jpg"

os.makedirs("img", exist_ok=True)

img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape


#linear grid
xs = np.linspace(0, w-1, NUM_X)
ys = np.linspace(0, h-1, NUM_Y) 

'''
#non-linear grid
xs = np.linspace(0, w-1, NUM_X)
ys_lin = np.linspace(0, 1, NUM_Y)
ys_nonlinear = 1 - (np.cos(ys_lin - 0.5))**2
ys_nonlinear = (ys_nonlinear - ys_nonlinear.min()) / (ys_nonlinear.max() - ys_nonlinear.min())
ys = ys_nonlinear * (h - 1)
'''

'''
#fibonacci grid
def fibonacci(n):
    seq = [1, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return np.array(seq)

fib_x = fibonacci(NUM_X)
fib_y = fibonacci(NUM_Y)

xs = np.cumsum(fib_x) / np.sum(fib_x) * (w-1)
ys = np.cumsum(fib_y) / np.sum(fib_y) * (h-1)
'''

grid_x, grid_y = np.meshgrid(xs, ys)
points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

n_total = len(points)
n_sample = int(n_total * FRAC)
index = np.random.choice(n_total, n_sample, replace=False)
points_reduced = points[index]

tri = Delaunay(points_reduced)

output = np.zeros_like(img_rgb)

for simplex in tri.simplices:
    pts = points_reduced[simplex].astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)

    mean_color = cv2.mean(img_rgb, mask=mask)[:3]
    mean_color = tuple(map(int, mean_color))

    cv2.fillConvexPoly(output, pts, mean_color)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"img/{IMG_PATH}_lowpoly_{timestamp}.jpg"
cv2.imwrite(filename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
