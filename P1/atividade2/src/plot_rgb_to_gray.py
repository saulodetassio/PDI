

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread, imsave
import os

# Caminhos
input_path = r'C:\PDI\P1\atividade2\data\input\paisagem.jpg'  # <- troque o nome conforme a imagem
output_path = r'C:\PDI\P1\atividade2\data\output\resultado_rgb_for_gray.png'

# Leitura da imagem
original = imread(input_path)
grayscale = rgb2gray(original)

# Criar figura com as duas imagens lado a lado
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
ax[1].axis('off')

fig.tight_layout()

# Salvar a figura resultante
fig.savefig(output_path)

# Exibir a figura
plt.show()

