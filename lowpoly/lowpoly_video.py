import cv2
import numpy as np
from scipy.spatial import Delaunay
import time
from datetime import datetime

# Settings
NUM_X_GRID = 10
NUM_Y_GRID = 6
TOTAL_POINTS = 600
EDGE_POINTS_RATIO = 0.6
SATURATION_FACTOR = 1.3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Output file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"lowpoly_{timestamp}.mp4"

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

num_test_frames = FPS
start = time.time()
for _ in range(num_test_frames):
    ret, frame = cap.read()
    if not ret:
        break
end = time.time()
real_fps = num_test_frames / (end - start)

if not cap.isOpened():
    raise RuntimeError("Failed to access webcam")

# Define codec & VideoWriter (MP4 / H.264)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter(output_filename, fourcc, real_fps, (FRAME_WIDTH, FRAME_HEIGHT))

print(f"Webcam opened. Recording to {output_filename}")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    if SATURATION_FACTOR != 1.0:
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 1] *= SATURATION_FACTOR
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_norm = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX)
    grad_norm = cv2.GaussianBlur(grad_norm, (7, 7), 0)

    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.vstack([xs.ravel(), ys.ravel()]).T
    probs = grad_norm.ravel() + 1e-6
    probs /= probs.sum()

    n_edge_points = int(TOTAL_POINTS * EDGE_POINTS_RATIO)
    edge_indices = np.random.choice(len(coords), size=n_edge_points, replace=False, p=probs)
    edge_points = coords[edge_indices]

    xs_grid = np.linspace(0, w-1, NUM_X_GRID)
    ys_grid = np.linspace(0, h-1, NUM_Y_GRID)
    grid_x, grid_y = np.meshgrid(xs_grid, ys_grid)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    points = np.vstack([grid_points, edge_points])

    try:
        tri = Delaunay(points)
    except Exception:
        continue

    output = np.zeros_like(img_rgb)

    for simplex in tri.simplices:
        pts = points[simplex].astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, pts, 1)
        mean_color = cv2.mean(img_rgb, mask=mask)[:3]
        cv2.fillConvexPoly(output, pts, tuple(map(int, mean_color)))

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow("Low Poly Webcam", output_bgr)
    out.write(output_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Closed. Video saved as {output_filename}")
