from skimage.morphology import skeletonize, thin, medial_axis
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.color import rgb2gray
import numpy as np
import os

# ==== CONFIGURAÇÃO DA IMAGEM ====
# Caminho da imagem
pasta_imagem = r"C:\PDI\P1\atividade3\imagens"
nome_imagem = "imagem_interessante.png"

# Caminho completo
caminho = os.path.join(pasta_imagem, nome_imagem)

# ==== LEITURA DA IMAGEM ====
# Carrega a imagem
imagem = io.imread(caminho)

# Converte para escala de cinza se for RGB
if imagem.ndim == 3:
    imagem = rgb2gray(imagem)

# Inverte se necessário (skeleton trabalha com fundo preto e objeto branco)
imagem = invert(imagem)

# Binariza (garante 0 e 1)
imagem_bin = imagem > 0.5

# ==== Skeletonization ====
skeleton = skeletonize(imagem_bin)
skeleton_lee = skeletonize(imagem_bin, method='lee')

# ==== Medial Axis ====
skel_medial, distance = medial_axis(imagem_bin, return_distance=True)
dist_on_skel = distance * skel_medial

# ==== Morphological thinning ====
thinned = thin(imagem_bin)
thinned_partial = thin(imagem_bin, max_num_iter=25)

# ==== Plotagem ====
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(imagem_bin, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].set_title('Skeleton (Zhang)')
ax[1].axis('off')

ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
ax[2].set_title('Skeleton (Lee)')
ax[2].axis('off')

ax[3].imshow(dist_on_skel, cmap='magma')
ax[3].contour(imagem_bin, [0.5], colors='w')
ax[3].set_title('Medial Axis')
ax[3].axis('off')

ax[4].imshow(thinned, cmap=plt.cm.gray)
ax[4].set_title('Thinned (Total)')
ax[4].axis('off')

ax[5].imshow(thinned_partial, cmap=plt.cm.gray)
ax[5].set_title('Thinned (Partial)')
ax[5].axis('off')

fig.tight_layout()
plt.show()
