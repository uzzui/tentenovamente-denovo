import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd, qr
from datetime import datetime
import os

#CONFIG
n = 7
N = 14444
lim = 0.5
cmap_lista = ['gray']
NUM_IMAGENS = 10

#DIR
os.makedirs("img", exist_ok=True)

for k_iter in range(NUM_IMAGENS):
    #TIMESTAMP
    agora = datetime.now()
    timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
    hora_str = agora.strftime("%H:%M:%S.%f")[:-3]
    data_str = agora.strftime("%d/%m/%Y")

    #GRID
    grid_pts = 144
    eps = 2
    x_lin = np.linspace(-lim-eps, lim+eps, grid_pts)
    y_lin = np.linspace(-lim-eps, lim+eps, grid_pts)
    X, Y = np.meshgrid(x_lin, y_lin)

    #SURFACE
    alpha = 0.2
    A = 10
    fx, fy = 3, 1
    E = np.sqrt(X**2 + Y**2)
    #Z = A * np.sin(20 * np.arctan2(Y, X)) / (E + alpha)
    #Z = A * np.sin(30 * np.arctan2(Y, X)) * np.exp(-0.2 * E)
    #Z = A * np.sin(fx * X + fy * Y) / (E + 0.05)
    #Z = A * np.exp(-E) * np.sign(np.sin(10 * E))
    Z = A * np.maximum(np.abs(X), np.abs(Y)) * np.sign(np.sin(5 * X + 5 * Y)) * np.cos(X*Y)
    #Z = A * (np.abs(np.sin(fx * X)) + np.abs(np.sin(fy * Y)))
    #Z = A * np.exp(-E) * np.sin(10 * np.arctan2(Y, X)) * np.cos(3 * E)

    pts3d_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts_embutidos = np.stack([np.pad(p, (0, n - 3)) for p in pts3d_flat], axis=1)

    #RANDOM MATRIX
    M = np.random.randn(n, n)
    U, S, VT = svd(M)
    pts_u = U @ pts_embutidos
    pts_vt = VT @ pts_embutidos
    conjuntos = [pts_vt[:3]] # aqui dá pra mudar pra pegar mais matrizes de transformação

    #PLOT
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    elevacoes = np.linspace(0, 90, 4)
    angulos = [(e, 30) for e in elevacoes]

    for i, (elev, azim) in enumerate(angulos):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_facecolor("black")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax.xaxis.line.set_color((0, 0, 0, 0))
        ax.yaxis.line.set_color((0, 0, 0, 0))
        ax.zaxis.line.set_color((0, 0, 0, 0))

        for j in range(len(conjuntos)):
            Xp = conjuntos[j][0].reshape(grid_pts, grid_pts)
            Yp = conjuntos[j][1].reshape(grid_pts, grid_pts)
            Zp = conjuntos[j][2].reshape(grid_pts, grid_pts)
            dist = np.sqrt(Xp**2 + Yp**2 + Zp**2)
            indice = (dist - dist.min()) / (dist.max() - dist.min())
            ax.plot_surface(Xp, Yp, Zp,
            facecolors=plt.colormaps[cmap_lista[j]](indice),
            rstride=1, cstride=1, antialiased=False, linewidth=0, shade=False
            )

    plt.subplots_adjust(
        left=0.05, right=0.95,
        top=0.95, bottom=0.08,
        wspace=0.02, hspace=0.02
    )

    filename = f"img/svd_surface3d_{timestamp}_{k_iter}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"[{k_iter + 1}/{NUM_IMAGENS}]: {filename}")


