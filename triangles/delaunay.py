import numpy as np
import cv2
from scipy.spatial import Delaunay


def eh_retangulo(pontos, tol_ang=10):
    c = np.mean(pontos, axis=0)
    ang = np.arctan2(pontos[:,1] - c[1], pontos[:,0] - c[0])
    pts = pontos[np.argsort(ang)]

    def angulo(a, b, c):
        v1 = a - b
        v2 = c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        cosang = np.clip(cosang, -1, 1)
        return np.degrees(np.arccos(cosang))

    angs = []
    for i in range(4):
        a = pts[(i-1) % 4]
        b = pts[i]
        c_ = pts[(i+1) % 4]
        angs.append(angulo(a, b, c_))

    return all(abs(a - 90) < tol_ang for a in angs)


def selecionar_pontos_com_distancia(mascara_inicial, xs, ys, min_dist):
    coords = []
    mascara = mascara_inicial.copy()

    indices = np.argwhere(mascara)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    for i, j in indices:
        p = np.array([xs[j], ys[i]])

        if len(coords) == 0:
            coords.append(p)
            continue

        # Verifica distância mínima
        dists = np.linalg.norm(np.array(coords) - p, axis=1)
        if np.all(dists >= min_dist):
            coords.append(p)
        else:
            mascara[i, j] = False

    return mascara


def gerar_composicao(
    nx=12,
    ny=12,
    proporcao_efetivos=(0.05, 0.2),
    n_blocos=1,
    largura=1080,
    altura=1350,
    margem=40,
    prob_remover_aresta=0.35,
    min_dist_pixels=70,          # DISTÂNCIA MÍNIMA ENTRE PONTOS
    seed=43,
    caminho_saida="composicao.png",
):

    rng = np.random.default_rng(seed)

    img = np.full((altura, largura, 3), 255, dtype=np.uint8)


    # Grid regular
    xs = np.linspace(margem, largura - margem, nx)
    ys = np.linspace(margem, altura - margem, ny)

    cinza = (210, 210, 210)
    #for x in xs:
    #    cv2.line(img, (int(x), int(ys[0])), (int(x), int(ys[-1])), cinza, 1)
    #for y in ys:
    #    cv2.line(img, (int(xs[0]), int(y)), (int(xs[-1]), int(y)), cinza, 1)

    # Seleção de pontos
    mascara = np.zeros((ny, nx), dtype=bool)

    # Borda
    mascara[0, :] = True
    mascara[-1, :] = True
    mascara[:, 0] = True
    mascara[:, -1] = True

    frac = rng.uniform(*proporcao_efetivos)
    internos = rng.random((ny - 2, nx - 2)) < frac
    mascara[1:-1, 1:-1] |= internos

    mascara = selecionar_pontos_com_distancia(mascara, xs, ys, min_dist_pixels)

    limites = [0, ny - 1]

    preto = (0, 0, 0)
    espessura = 5

    # Moldura
    cv2.rectangle(
        img,
        (int(xs[0]), int(ys[0])),
        (int(xs[-1]), int(ys[-1])),
        preto,
        espessura,
    )

    def coord(i, j):
        return np.array([xs[j], ys[i]])

    indices = np.argwhere(mascara)
    pts = np.array([coord(i, j) for i, j in indices])

    tri = Delaunay(pts)

    contagem_arestas = {}
    for simplex in tri.simplices:
        for a, b in [(0,1),(1,2),(2,0)]:
            i = simplex[a]
            j = simplex[b]
            e = tuple(sorted((i,j)))
            contagem_arestas[e] = contagem_arestas.get(e,0) + 1

    arestas_para_desenhar = set(contagem_arestas.keys())

    for e, count in contagem_arestas.items():
        if count == 2 and rng.random() < prob_remover_aresta:

            tri_ids = [
                s for s in tri.simplices
                if set(e).issubset(s)
            ]

            pontos_quad_idx = []
            for t in tri_ids:
                for idx in t:
                    if idx not in pontos_quad_idx:
                        pontos_quad_idx.append(idx)

            if len(pontos_quad_idx) == 4:
                pontos_quad = np.array([pts[i] for i in pontos_quad_idx])

                if not eh_retangulo(pontos_quad):
                    continue

            arestas_para_desenhar.discard(e)
    for i_local, j_local in arestas_para_desenhar:
        p1 = tuple(np.round(pts[i_local]).astype(int))
        p2 = tuple(np.round(pts[j_local]).astype(int))
        cv2.line(img, p1, p2, preto, espessura)

    cv2.imwrite(caminho_saida, img)
    print(f"Imagem gerada: {caminho_saida}")



# Execução
if __name__ == "__main__":
    gerar_composicao()
