import cv2

# Carregar imagem colorida
img_rgb = cv2.imread('paisagem.jpg')

# Converter para escala de cinza (3 métodos comuns)
gray_avg = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Média simples R+G+B/3
gray_weighted = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Peso luminância (0.299R + 0.587G + 0.114B)
gray_desaturation = (img_rgb.max(axis=2) + img_rgb.min(axis=2)) / 2  # Dessaturação

# Salvar resultados
cv2.imwrite('gray_average.jpg', gray_avg)
cv2.imwrite('gray_weighted.jpg', gray_weighted)
cv2.imwrite('gray_desaturation.jpg', gray_desaturation)