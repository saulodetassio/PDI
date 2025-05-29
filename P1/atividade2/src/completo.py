import os
import matplotlib.pyplot as plt
from skimage import io, color, exposure
from skimage.transform import resize
import numpy as np

# Caminhos
input_path = r'C:\PDI\P1\atividade2\data\input\sol_2.jpg'
output_dir = r'C:\PDI\P1\atividade2\data\output'
os.makedirs(output_dir, exist_ok=True)

# Leitura da imagem
original = io.imread(input_path)

# Redimensionamento para facilitar o processamento
original_resized = resize(original, (512, 512), anti_aliasing=True)

# Conversão para escala de cinza
gray = color.rgb2gray(original_resized)

# Conversão para HSV
hsv = color.rgb2hsv(original_resized)

# Equalização global do histograma
equalized_global = exposure.equalize_hist(gray)

# Equalização local do histograma
equalized_local = exposure.equalize_adapthist(gray, clip_limit=0.03)

# Correspondência de histograma (usando a própria imagem original como referência)
matched = exposure.match_histograms(gray, gray, channel_axis=None)

# Plotagem dos resultados
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(original_resized)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Escala de Cinza')
axes[1].axis('off')

axes[2].imshow(hsv)
axes[2].set_title('HSV')
axes[2].axis('off')

axes[3].imshow(equalized_global, cmap='gray')
axes[3].set_title('Equalização Global')
axes[3].axis('off')

axes[4].imshow(equalized_local, cmap='gray')
axes[4].set_title('Equalização Local')
axes[4].axis('off')

axes[5].imshow(matched, cmap='gray')
axes[5].set_title('Correspondência de Histograma')
axes[5].axis('off')

plt.tight_layout()
output_path = os.path.join(output_dir, 'analise_sol.png')
plt.savefig(output_path)
plt.show()
