import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import (disk, rectangle, line, polygon)
from skimage.io import imsave
import os

# Criar uma imagem vazia (preta)
img = np.zeros((300, 300), dtype=np.uint8)

# 🔵 Desenhar dois círculos
rr, cc = disk((80, 80), 40)
img[rr, cc] = 255

rr, cc = disk((220, 80), 40)
img[rr, cc] = 255

# 🔲 Desenhar um retângulo
rr, cc = rectangle(start=(60, 180), end=(240, 260))
img[rr, cc] = 255

# 🔺 Desenhar um triângulo
r = np.array([180, 250, 110])
c = np.array([50, 150, 150])
rr, cc = polygon(r, c)
img[rr, cc] = 255

# ➕ Desenhar linhas cruzadas
rr, cc = line(50, 50, 250, 250)
img[rr, cc] = 255

rr, cc = line(50, 250, 250, 50)
img[rr, cc] = 255

# 🔤 Adicionar um pequeno texto (simulado como um quadrado para simplificação)
rr, cc = rectangle(start=(130, 130), end=(170, 170))
img[rr, cc] = 0  # Recorte para simular texto no meio do retângulo

# ======= Salvar =======
pasta = r"C:\PDI\P1\atividade3\imagens"
os.makedirs(pasta, exist_ok=True)

caminho = os.path.join(pasta, "imagem_interessante.png")
imsave(caminho, img)

# Mostrar para conferência
plt.imshow(img, cmap='gray')
plt.title('Imagem Binária Gerada')
plt.axis('off')
plt.show()

print(f"Imagem salva em {caminho}")
