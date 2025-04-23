import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from random import randrange

#dimensiono a imagem
figsize = (10,10)

#leio a imagem e converto de BGR para RGB
rgb_l = cv2.cvtColor(cv2.imread("bola_left.png"), cv2.COLOR_BGR2RGB)
#converto a imagem para escala de cinza
gray_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2GRAY)
#leio a imagem e converto de BGR para RGB
rgb_r = cv2.cvtColor(cv2.imread("bola_right.png"), cv2.COLOR_BGR2RGB)
#converto a imagem para escala de cinza
gray_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2GRAY)


#aplica o SIFT para extrair características da imagem
feature_extractor = cv2.SIFT_create()

#Faz a detecção e computação das caracteríticas da imagem usando o método SIFT
kp_l, desc_l = feature_extractor.detectAndCompute(gray_l, None)
kp_r, desc_r = feature_extractor.detectAndCompute(gray_r, None)

#desenha os keypoins recebendo as caracterísitcas estraídas do método SIFT
test = cv2.drawKeypoints(rgb_l, kp_l, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
test2 = cv2.drawKeypoints(rgb_r, kp_r, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plota a figura com o tamanho figsize
plt.figure(figsize=figsize)
#plota o 'test'
plt.imshow(test)
#imprime o titulo
plt.title("Keypoints")
#mostra de fato a imagem
plt.show()


bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_l, desc_r, k=2)

#apply ratio test
good_match = []

for m in matches:
    if m[0].distance/m[1].distance < 0.5:
        good_match.append(m)

good_match_arr = np.asarray(good_match)

#show only 30 matches
im_matches = cv2.drawMatchesKnn(rgb_l, kp_l, rgb_r, kp_r, good_match[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#plota a figura com o tamanho figsize
plt.figure(figsize=(20,20))
#plota a figura
plt.imshow(im_matches)
#imprime o titulo
plt.title("Keypoints matches")
#mostra de fato a imagem
plt.show()


good_kp_l = np.array([kp_l[m.queryIdx].pt for m in good_match_arr[:, 0]]).reshape(-1,1,2)
good_kp_r = np.array([kp_r[m.trainIdx].pt for m in good_match_arr[:, 0]]).reshape(-1,1,2)
H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)
print(H)

rgb_r_warped = cv2.warpPerspective(rgb_r, H, (rgb_l.shape[1] + rgb_r.shape[1], rgb_l.shape[0]))
rgb_r_warped[0:rgb_l.shape[0], 0:rgb_l.shape[1]] = rgb_l

plt.figure(figsize=figsize)
plt.imshow(rgb_r_warped)
plt.title("Naive warping")
plt.show()