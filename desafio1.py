# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

def binarizacao_otsu(img):
  print("Binarização")
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

  gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

  kernel_fundo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
  fundo = cv2.morphologyEx(gray_blur, cv2.MORPH_CLOSE, kernel_fundo)
  corrigida = cv2.divide(gray_blur, fundo, scale=255)

  _, binaria = cv2.threshold(corrigida,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # remover canto inferior direito
  altura, largura = binaria.shape
  binaria[altura-70:altura,largura-70:largura] = 0

  #mostrar a imagem original e a do otsu
  plt.figure(figsize=(15,10))

  plt.subplot(1,3,1)
  plt.imshow(img_rgb)
  plt.axis("off")
  plt.title("Imagem - Original")

  plt.subplot(1,3,2)
  plt.imshow(corrigida)
  plt.axis("off")
  plt.title("Imagem - corrigida")

  plt.subplot(1,3,3)
  plt.imshow(binaria)
  plt.axis("off")
  plt.title("Imagem - Binarizada binaria")

  plt.show()

  return binaria

def morfológicas(binaria):
  print("Morfológicas")
  kernel = np.ones((3, 3), np.uint8)
  limpa  = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)

  eroded = cv2.erode(
      limpa,
      kernel,
      iterations=2
  )

  eroded_close = cv2.dilate(eroded, kernel, iterations=2)
  eroded_close = cv2.morphologyEx(eroded_close, cv2.MORPH_CLOSE, kernel, iterations=2)

  plt.figure(figsize=(15,10))

  plt.subplot(1,3,1)
  plt.imshow(limpa)
  plt.axis("off")
  plt.title("Limpa")

  plt.subplot(1,3,2)
  plt.imshow(eroded)
  plt.axis("off")
  plt.title("eroded")

  plt.subplot(1,3,3)
  plt.imshow(eroded_close)
  plt.axis("off")
  plt.title("eroded_close")

  return limpa, eroded, eroded_close

#distance transform e watershed
def watershed(img, eroded_close):
  print("Watershed")
  dist = cv2.distanceTransform(eroded_close, cv2.DIST_L2, 5)

  _, sure_fg = cv2.threshold(
      dist,
      0.35 * dist.max(),
      255,
      cv2.THRESH_BINARY
  )

  sure_fg = np.uint8(sure_fg)

  plt.figure(figsize=(8,8))
  plt.imshow(sure_fg, cmap="gray")
  plt.axis("off")
  plt.title("Sure foreground - sementes do watershed")
  plt.show()

  num_labels, markers = cv2.connectedComponents(sure_fg)

  markers = markers + 1

  unknown = cv2.subtract(eroded_close, sure_fg)
  markers[unknown == 255] = 0

  img_watershed = img.copy()
  markers = cv2.watershed(cv2.cvtColor(img_watershed, cv2.COLOR_RGB2BGR), markers)

  return sure_fg, markers

def labels(binaria, limpa, eroded, eroded_close, sure_fg, markers):
  print("Labels")
  num_labels_binaria, _ = cv2.connectedComponents(binaria)
  print("binaria:", num_labels_binaria-1)

  num_labels_limpa, _ = cv2.connectedComponents(limpa)
  print("limpa:", num_labels_limpa-1)

  num_labels_eroded, _ = cv2.connectedComponents(eroded)
  print("eroded:", num_labels_eroded-1)

  num_labels_eroded_close, _ = cv2.connectedComponents(eroded_close)
  print("eroded_close:", num_labels_eroded_close-1)

  num_labels_sure_fg, _ = cv2.connectedComponents(sure_fg)
  print("sure_fg:", num_labels_sure_fg-1)

  # num_labels, _ = cv2.connectedComponents(markers)
  # print("markers:", num_labels-1)

################################
def contornos(eroded_close, markers):
  print("Contornos")
  contours_debug, _ = cv2.findContours(eroded_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  area_threshold = 500.0
  area_threshold_watershed = 250.0

  ###########################

  valid_contours_normal = [
      cnt for cnt in contours_debug
      if cv2.contourArea(cnt) > area_threshold
  ]

  valid_contours_watershed = []
  #########################

  for label in np.unique(markers):
      if label <= 1:
          continue

      mask = np.zeros(eroded_close.shape, dtype="uint8")
      mask[markers == label] = 255

      contours_label, _ = cv2.findContours(
          mask,
          cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE
      )

      for cnt in contours_label:
          area = cv2.contourArea(cnt)

          if area > area_threshold_watershed:
              valid_contours_watershed.append(cnt)

  areas_normais = [cv2.contourArea(cnt) for cnt in valid_contours_normal]
  area_mediana = np.median(areas_normais) if len(areas_normais) > 0 else 0

  valid_contours = []
  usou_watershed = False

  for cnt_normal in valid_contours_normal:
      x, y, w, h = cv2.boundingRect(cnt_normal)
      area_normal = cv2.contourArea(cnt_normal)

      candidatos_watershed = []

      for cnt_watershed in valid_contours_watershed:
          xw, yw, ww, hw = cv2.boundingRect(cnt_watershed)

          inter_x1 = max(x, xw)
          inter_y1 = max(y, yw)
          inter_x2 = min(x + w, xw + ww)
          inter_y2 = min(y + h, yw + hw)

          inter_w = max(0, inter_x2 - inter_x1)
          inter_h = max(0, inter_y2 - inter_y1)
          inter_area = inter_w * inter_h

          area_watershed_box = ww * hw
          sobrepoe_contorno_normal = (
              area_watershed_box > 0 and
              inter_area / area_watershed_box > 0.5
          )

          if sobrepoe_contorno_normal:
              candidatos_watershed.append(cnt_watershed)

      contorno_grande = area_mediana > 0 and area_normal > 1.4 * area_mediana

      if contorno_grande:
          print(
              "Contorno grande:",
              f"area={area_normal:.1f}",
              f"mediana={area_mediana:.1f}",
              f"candidatos_watershed={len(candidatos_watershed)}"
          )
      print("contorno grande: ", contorno_grande, "candidatos_watershed: ", len(candidatos_watershed))
      if contorno_grande and len(candidatos_watershed) > 1:
          valid_contours.extend(candidatos_watershed)
          usou_watershed = True
      else:
          valid_contours.append(cnt_normal)

  metodo_usado = "normal + watershed pontual" if usou_watershed else "normal"

  print("Metodo usado:", metodo_usado)

  contour_count = len(valid_contours)

  print(f"Number of contours above threshold: {contour_count}")

  return valid_contours, contours_debug, area_threshold

def contornos_debug(img, contours_debug, area_threshold):
  print("Contornos debug")
  img_todos = img.copy()

  for i, cnt in enumerate(contours_debug):
      x, y, w, h = cv2.boundingRect(cnt)
      area = cv2.contourArea(cnt)

      if area > area_threshold:
          cor = (0, 255, 0)   # verde: contado
      else:
          cor = (255, 0, 0)   # vermelho: descartado

      cv2.rectangle(img_todos, (x, y), (x + w, y + h), cor, 2)
      cv2.putText(
          img_todos,
          f"{i+1}: {int(area)}",
          (x, y - 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          cor,
          1
      )

  plt.figure(figsize=(10,10))
  plt.imshow(img_todos)
  plt.axis("off")
  plt.title("Contornos: verde contado, vermelho descartado")
  plt.show()

def contornos_validos(img, valid_contours):
  print("Contornos válidos")
  print("Quantidade de valid_contours:", len(valid_contours))

  for i, cnt in enumerate(valid_contours):
      x, y, w, h = cv2.boundingRect(cnt)
      area = cv2.contourArea(cnt)

      print(
          f"{i+1}: x={x}, y={y}, w={w}, h={h}, area={area}"
      )

  img_contours = img.copy()

  for i, cnt in enumerate(valid_contours):

      x, y, w, h = cv2.boundingRect(cnt)

      # caixa verde
      cv2.rectangle(
          img_contours,
          (x, y),
          (x + w, y + h),
          (0, 255, 0),
          2
      )

      # número vermelho
      cv2.putText(
          img_contours,
          str(i + 1),
          (x, y - 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 0, 0),
          2
      )

  plt.figure(figsize=(10,10))
  plt.imshow(img_contours)
  plt.axis("off")
  plt.title("Objetos contados")
  plt.show()

  return len(valid_contours)

import cv2
real = [8, 10, 10, 4]

detectado = []
#pegar images dos arquivos aqui do colab
imagens = [
    'images/img1.jpg',
    'images/img2.jpg',
    'images/img3.jpg',
    'images/img4.jpg',
    'images/img5.jpg'
]

for imagem in imagens:
    img = cv2.imread(imagem)
    # --- Pipeline de processamento de imagem ---
    binaria_result = binarizacao_otsu(img)
    limpa_result, eroded_result, eroded_close_result = morfológicas(binaria_result)
    sure_fg_result, markers_result = watershed(img, eroded_close_result)
    labels(binaria_result, limpa_result, eroded_result, eroded_close_result, sure_fg_result, markers_result)
    valid_contours_result, contours_debug_result, area_threshold_result = contornos(eroded_close_result, markers_result)
    contornos_debug(img, contours_debug_result, area_threshold_result)
    detectado_img = contornos_validos(img, valid_contours_result)

    detectado.append(detectado_img)

print("detectado: ", detectado)

#Métricas:

import numpy as np

real = [8, 1, 4, 2, 10]

erros_abs = np.abs(np.array(real) - np.array(detectado))
erros_diff = np.array(detectado) - np.array(real) # Para calcular o Bias, sem abs

mae = np.mean(erros_abs)

acertos = np.sum(np.array(real) == np.array(detectado))

acuracia = (acertos /len(real)) * 100

erro_percentual = np.mean(erros_abs / np.array(real)) * 100

# Novas métricas
rmse = np.sqrt(np.mean(erros_diff**2))
bias = np.mean(erros_diff)

print("========== RESULTADOS ==========")
print("MAE - Erro Médio Absoluto:", mae)
print("RMSE - Raiz Quadrada do Erro Médio Quadrático:", round(rmse, 2))
print("Bias - Tendência de Contagem (média do erro):", round(bias, 2))
print("Acurácia:", round(acuracia,2), "%")
print("Erro percentual médio:", round(erro_percentual,2),"%")
