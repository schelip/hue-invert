import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def invert_hue(img: cv.Mat, m: int, x: int) -> cv.Mat:
    """
    Inverte o hue da imagem dentro de uma faixa especificada.

    Args:
        img (cv.Mat): A imagem de entrada no formato OpenCV.
        m (int): O valor da matiz central da faixa.
        x (int): A largura da faixa em unidades de matiz.

    Returns:
        cv.Mat: A imagem com a matiz invertida na faixa especificada.
    """
    print(f"Processando faixa [{m - x}, {m + x}] (m = {m}, x = {x})")
    img = img.astype(np.float32)/255
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    interval_filter = (h >= max(m - x, 0)) & (h <= min(m + x, 360))
    h[interval_filter] = (180 + h[interval_filter]) % 360
    hsv_inv = cv.merge([h, s, v])
    img_inv = cv.cvtColor(hsv_inv, cv.COLOR_HSV2RGB)
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

def show_image(img: cv.Mat, title: str) -> None:
    """
    Exibe uma imagem com um título.

    Args:
        img (cv.Mat): A imagem a ser exibida no formato OpenCV.
        title (str): O título a ser exibido acima da imagem.
    """
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def main():
    """
    Função principal que processa a linha de comando e executa as operações desejadas.
    """
    if len(sys.argv) < 4:
        print("Uso: py matizes-hsv.py CAMINHO-IMAGEM [MATIZ FATOR | [--save-anim | -a] CAMINHO]")
        exit()

    try:
        img = cv.imread(sys.argv[1])
    except:
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
    show_image(img_inv, "Imagem Manipulada")

if __name__ == '__main__':
    main()
