import os
import matplotlib.pyplot as plt
from skimage import io, color, exposure
from skimage.transform import resize
import numpy as np

# --- Configuração de Caminhos ---
input_path = r'C:\PDI\P1\atividade2\data\input\sol_2.jpg'
output_dir = r'C:\PDI\P1\atividade2\data\output'

os.makedirs(output_dir, exist_ok=True)

# --- Carregamento e Pré-processamento da Imagem ---
original = io.imread(input_path)

# Redimensiona a imagem original.
# O resize mantém o tipo float se a imagem original for float, ou converte se necessário.
original_resized = resize(original, (512, 512), anti_aliasing=True)

# Converte para escala de cinza (normalmente resulta em float).
gray = color.rgb2gray(original_resized)

# Converte para HSV (normalmente resulta em float).
hsv = color.rgb2hsv(original_resized)

# --- Aplicação das Técnicas de Processamento de Imagens ---
equalized_global = exposure.equalize_hist(gray)
equalized_local = exposure.equalize_adapthist(gray, clip_limit=0.03)
matched = exposure.match_histograms(gray, gray, channel_axis=None)

# --- Função Auxiliar para Plotar e Salvar Imagens e Histogramas ---
def plot_and_save(image, title, cmap, filename_prefix, output_dir, is_color=False):
    """
    Plota uma imagem e seu histograma, e salva-os como arquivos separados.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)

    # Plot the image
    if is_color:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap=cmap)
    axes[0].set_title(f'{title} Image')
    axes[0].axis('off')

    # Plot the histogram
    if is_color and image.ndim == 3:
        hist_image = color.rgb2gray(image)
    else:
        hist_image = image

    hist, bin_centers = exposure.histogram(hist_image)
    axes[1].plot(bin_centers, hist)
    axes[1].set_title(f'{title} Histogram')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Number of Pixels')
    axes[1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the combined image and histogram plot
    combined_plot_path = os.path.join(output_dir, f'{filename_prefix}_image_and_hist.png')
    plt.savefig(combined_plot_path)
    plt.close(fig)

    # --- CORREÇÃO AQUI: Converter imagens coloridas para uint8 antes de salvar ---
    separate_image_path = os.path.join(output_dir, f'{filename_prefix}_image.png')
    if is_color:
        # Escala os valores de 0-1 para 0-255 e converte para uint8.
        # Usa np.clip para garantir que os valores estejam dentro do range antes da conversão.
        io.imsave(separate_image_path, (np.clip(image, 0, 1) * 255).astype(np.uint8))
    else:
        # Para imagens em escala de cinza (que já são float 0-1), a conversão já estava correta.
        io.imsave(separate_image_path, (image * 255).astype(np.uint8))


# --- Plotting and Saving Results ---
images_to_process = [
    (original_resized, 'Original', None, 'original', True),
    (gray, 'Escala de Cinza', 'gray', 'grayscale', False),
    (hsv, 'HSV', None, 'hsv', True), # Este era o principal causador do erro ao salvar
    (equalized_global, 'Equalização Global', 'gray', 'equalized_global', False),
    (equalized_local, 'Equalização Local', 'gray', 'equalized_local', False),
    (matched, 'Correspondência de Histograma', 'gray', 'matched_hist', False)
]

for img, title, cmap, filename_prefix, is_color in images_to_process:
    plot_and_save(img, title, cmap, filename_prefix, output_dir, is_color)

print(f"Todas as imagens processadas e seus histogramas foram salvos na pasta: {output_dir}")