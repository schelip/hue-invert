# ALTERAÇÃO DE FAIXA DE MATIZES EM UMA IMAGEM NO SISTEMA HSV
# 
# RA117306 | Felipe Gabriel Comin Scheffel
# RA117741 | Douglas Kenji Sakakibara

import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150  
plt.ioff()

def invert_hue(a: cv.Mat, m: int, x: int) -> cv.Mat:
    """
    Inverte o hue da imagem dentro de uma faixa especificada.

    Args:
        a (cv.Mat): A imagem de entrada no formato OpenCV.
        m (int): O valor da matiz central da faixa.
        x (int): A largura da faixa em unidades de matiz.

    Returns:
        cv.Mat: A imagem com a matiz invertida na faixa especificada.
    """
    print(f"Processando faixa [{m - x}, {m + x}] (m = {m}, x = {x})")
    a = a.astype(np.float32) / 255                                     # Normalizar valores da matriz para faixa [0, 1]
    hsv = cv.cvtColor(a, cv.COLOR_BGR2HSV)                             # Converter do sistema de cor padrão (BGR) para HSV
    h, s, v = cv.split(hsv)                                            # Separar bandas de Hue, Saturation e Value em diferentes matrizes
    interval_filter = (h >= max(m - x, 0)) & (h <= min(m + x, 360))    # Filtro para faixa de inversão
    h[interval_filter] = (180 + h[interval_filter]) % 360              # Inverter valores para elementos que o filtro abrage
    hsv_inv = cv.merge([h, s, v])                                      # Reunificar matrizes do HSV
    img_inv = cv.cvtColor(hsv_inv, cv.COLOR_HSV2RGB)                   # Converter de HSV para RGB
    return img_inv


def save_animation(img: cv.Mat, path: str) -> None:
    """
    Cria uma animação que varia o hue da imagem e a salva em um arquivo.

    Args:
        img (cv.Mat): A imagem de entrada no formato OpenCV.
        path (str): O caminho do arquivo de saída para a animação.
    """
    def updatefig(*_):
        nonlocal m, width, im
        if m > (360 - width - 1): m = width
        else: m += 10
        im.set_array(invert_hue(img, m, width))
        return im

    fig = plt.figure()
    width = 20
    frames = (((360 - width) - width) // 10) + 1

    im = plt.imshow(img, animated=True)

    m = width - 10 # Preparando para a primeira iteração, que soma 10
    anim = animation.FuncAnimation(fig, updatefig, init_func=lambda: im, frames=frames, interval=200)
    print("Salvando animação...")
    anim.save(path)


def plot_images(img1: cv.Mat, img2: cv.Mat) -> None:
    """
    Recebe duas imagens e exibe ambas lado a lado em um único plot. Ambas devem estar no modelo RGB.

    Args:
        img (cv.Mat): A imagem a ser exibida no formato OpenCV.
        img_inv (cv.Mat): A imagem a ser exibida no formato OpenCV.
    """
    fig, axes = plt.subplots(1, 2, dpi=200)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()


def main():
    """
    Função principal que processa a linha de comando e executa as operações desejadas.
    """
    if len(sys.argv) < 4:
        print("Uso: py matizes-hsv.py CAMINHO-IMAGEM [MATIZ FATOR | [--save-anim | -a] CAMINHO]")
        exit()

    img = cv.imread(sys.argv[1])

    if img is None:
        print("Erro ao abrir imagem")
        exit()

    if sys.argv[2] in ['--save-anim', '-a']:
        path = sys.argv[3] if len(sys.argv) == 4 else None
        save_animation(img, path)
        return

    m = int(sys.argv[2])
    if m < 0 or m > 359:
        print("Valor inválido para m (deve satisfazer 0 <= m < 360)")
        exit()

    x = int(sys.argv[3])
    img_inv = invert_hue(img, m, x)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plot_images(img, img_inv)

if __name__ == '__main__':
    main()
