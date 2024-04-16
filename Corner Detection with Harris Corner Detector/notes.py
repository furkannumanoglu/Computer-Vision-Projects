# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy
import time

# You can use this code block (or add new code blocks here) to implement shared functionality.
images = ['chessboard.png', 'chessboard-rotated.png', 'lab.png', 'lab-rotated.png', 'tree.png', 'tree-rotated.png']


# images = ['tree.png', 'tree-rotated.png']
# images = ['chessboard.png', 'chessboard-rotated.png']
# images = ['tree.png']


def calculateChannelGradients(image):
    shape_0, shape_1 = image.shape[0], image.shape[1]
    E_array = np.zeros((shape_0, shape_1, 3))  # x x 3 şeklinde bir matris oluştur

    # calculating over R G B seperately
    for channel in range(3):
        for i in range(2, shape_0 - 2):
            for j in range(2, shape_1 - 2):
                base_window = image[i - 1:i + 2, j - 1:j + 2, channel]

                windows = [
                    image[i:i + 3, j:j + 3, channel],
                    image[i:i + 3, j - 1:j + 2, channel],
                    image[i:i + 3, j - 2:j + 1, channel],
                    image[i - 1:i + 2, j:j + 3, channel],
                    image[i - 1:i + 2, j - 2:j + 1, channel],
                    image[i - 2:i + 1, j:j + 3, channel],
                    image[i - 2:i + 1, j - 1:j + 2, channel],
                    image[i - 2:i + 1, j - 2:j + 1, channel]
                ]

                # broadcasting ile toplam
                sum_squares = np.sum((base_window - np.array(windows)) ** 2, axis=(1, 2))
                E_array[i][j][channel] = np.sum(sum_squares)

    return E_array


# grayscale E hesaplıyor sadece
def calculateWithGray(image):
    shape_0, shape_1 = image.shape[0], image.shape[1]
    E_array = np.zeros((shape_0, shape_1))  # x x  şeklinde bir matris oluştur

    for i in range(2, shape_0 - 2):
        for j in range(2, shape_1 - 2):
            base_window = image[i - 1:i + 2, j - 1:j + 2]
            windows = [
                image[i:i + 3, j:j + 3],
                image[i:i + 3, j - 1:j + 2],
                image[i:i + 3, j - 2:j + 1],
                image[i - 1:i + 2, j:j + 3],
                image[i - 1:i + 2, j - 2:j + 1],
                image[i - 2:i + 1, j:j + 3],
                image[i - 2:i + 1, j - 1:j + 2],
                image[i - 2:i + 1, j - 2:j + 1]
            ]
            # broadcasting ile toplam
            sum_squares = np.sum((base_window - np.array(windows)) ** 2, axis=(1, 2))
            E_array[i][j] = np.sum(sum_squares)
    return E_array


def nonMaxSuppression(E_array, shape_0, shape_1):
    E_array = np.abs(E_array)  # taylor yaklaşımında negatif ifade gelebiliyor bunun için yaptım.

    temp = np.ones((shape_0, shape_1))

    for i in range(1, shape_0 - 1):  # tüm pikselleri dönüyorum
        for j in range(1, shape_1 - 1):
            window = E_array[i - 1:i + 2, j - 1:j + 2]  # 3x3 windowum var
            max_value = np.max(window)

            if (max_value == 0):
                continue

            max_indices = np.where(window == max_value)
            non_max_indices = np.where(window != max_value)
            max_indices = np.column_stack((max_indices[0], max_indices[1]))
            non_max_indices = np.column_stack((non_max_indices[0], non_max_indices[1]))

            # window_indices = np.append(max_indices, non_max_indices)
            window_indices = np.concatenate((max_indices, non_max_indices), axis=0)

            idx = -1
            idy = -1

            if E_array[i][j] == max_value and len(max_indices) > 1:  # baktığım eğer max ise ama başka maxlar da varsa
                for index in max_indices:
                    if temp[i + index[0] - 1][j + index[1] - 1] == 1:
                        idx = index[0]
                        idy = index[1]
                        break

                for index in max_indices:
                    if idx == index[0] and idy == index[1]:
                        continue
                    else:
                        temp[i + index[0] - 1][j + index[1] - 1] = 0

            elif E_array[i][j] == max_value and len(
                    max_indices) == 1:  # baktığım eğer max ise ama başka maxlar da varsa
                for index in window_indices:
                    if i + index[0] - 1 == i and j + index[1] - 1 == j:
                        temp[i + index[0] - 1][j + index[1] - 1] = 1
                    else:
                        temp[i + index[0] - 1][j + index[1] - 1] = 0

            elif E_array[i][j] != max_value:  # baktığım eğer max ise ama başka maxlar da varsa
                temp[i][j] = 0

    temp = np.multiply(temp, E_array)
    return temp
def fastHarris(img):
    img = cv2.imread(
        image)  # burda image ı okurken direkt ikinci parametreye 0 girip gray okursam bulduğu köşeler değişiyor mutlaka sor birine!
    # (yine köşe buluyor sadece konumsal değişiklik var)
    # img = img = cv2.imread(image,0) dan bahsediyorum

    result_img = img.copy()

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    shape_0, shape_1 = gray_image.shape[0], gray_image.shape[1]

    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    """dx = np.array([[-100, 0, 100], [-200, 0, 200], [-100, 0, 100]])
    dy = np.array([[-100, -200, -100], [0, 0, 0], [100, 200, 100]])"""

    """Ix = cv2.filter2D(gray_image, cv2.CV_64F, dx)
    Iy = cv2.filter2D(gray_image, cv2.CV_64F, dy)"""

    Ix, Iy = np.gradient(gray_image)

    """Ix2 = np.matmul(Ix, Ix)
    Iy2 = np.matmul(Iy, Iy)
    Ixy = np.matmul(Ix,Iy)
    Iyx = np.matmul(Iy, Ix)"""
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix * Iy

    E_array = np.zeros_like(gray_image, dtype=np.float64)

    print(Ix[18][15])
    print(Iy[18][15])
    print(Ix2[18][15])
    print(Iy2[18][15])
    print(Ixy[18][15])

    print(Ixy[79][199])
    """offset = int( 240 / 2 )
    for y in range(offset, 240-offset):
        for x in range(offset, 240-offset):
            Sx2 = np.sum(Ix2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(Iy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])

            #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det=np.linalg.det(H)
            tr=np.matrix.trace(H)
            R=det-0.04*(tr**2)
            E_array[y-offset, x-offset] = R"""

    for i in range(1, shape_0 - 1):
        for j in range(1, shape_1 - 1):  # her piksel için dönüyorum
            M_Ix2 = 0
            M_Iy2 = 0
            M_Ixy = 0
            M_Iyx = 0

            for k in range(-1, 2):
                for m in range(-1, 2):
                    M_Ix2 += Ix2[i + k][j + m]
                    M_Iy2 += Iy2[i + k][j + m]
                    M_Ixy += Ixy[i + k][j + m]
                    M_Iyx += Ixy[i + k][j + m]

            M = np.array([[M_Ix2, M_Ixy], [M_Iyx, M_Iy2]])
            # M = np.array([[Ix2[i][j], Ixy[i][j]], [Iyx[i][j], Iy2[i][j]]])
            # M = np.column_stack(M)

            trace = np.matrix.trace(M)
            determinant = np.linalg.det(M)

            Q = determinant - 0.04 * (trace ** 2)  # 0.06 değişebilir!!
            if Q > 0:
                E_array[i][j] = Q
                if i == 18 and j == 15:
                    print(Q)

    # E_array = Ix2*Iy2 - (Ixy*Ixy) - 0.04*(Ix2 + Iy2)*(Ix2 + Iy2)
    E_array = nonMaxSuppression(E_array, shape_0, shape_1)
    # nonmaximum supression kısmı!!

    sorted_array = np.sort(E_array, axis=None)  # sortlandı tek bi liste olarak döndü

    high_array = sorted_array[-10:]

    # np.isin eşit olan tüm gradientleri alacağı için sıkıntı. eşit olan tüm gradientleri true'ya çevirir. en sonda 10 elemanlı elde edemem.
    high_index_array = np.where(np.isin(E_array, high_array))

    high_index_array = np.column_stack((high_index_array[0], high_index_array[1]))

    high_index_array = np.unique(high_index_array, axis=0)

    high_index_array = high_index_array[
                       :10]  # chessboard gibi aynı E nin gelebileceği örüntü olan resimlerde, 10 köşeden fazlasını unique kabul etmemesi için

    print(high_index_array)

    for index in high_index_array:
        cv2.circle(result_img, (index[1], index[0]), 2, (0, 0, 255), -1)

    # return fortest
    return result_img


for i, image in enumerate(images):
    result_img = fastHarris(image)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
