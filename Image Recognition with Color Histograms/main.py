# This is CEng 483 THE1.

# Furkan Numanoglu - 2448710

# November 2023
import cv2
import numpy as np


def l1NormalizationPcHistogram(histogram):
    n = np.linalg.norm(histogram, 1)
    if n == 0:
        return histogram
    histlen = len(histogram)
    for i in range(histlen):
        histogram[i] /= n
    return histogram


def l1Normalization3DHistogram(histogram):
    l1 = histogram.shape[0]
    for i in range(l1):
        for j in range(l1):
            for k in range(l1):
                histogram[i][j][k] /= 9216
    return histogram


def l1Normalization3DHistogramGrid(histogram, grid):
    n = 9216 / grid
    l1 = histogram.shape[0]
    for i in range(l1):
        for j in range(l1):
            for k in range(l1):
                histogram[i][j][k] /= n
    return histogram


def l1NormalizationImage(image):
    normalizedImage = np.zeros((96, 96, 3), float)
    for i in range(96):
        for j in range(96):
            for k in range(3):
                normalizedImage[i][j][k] = image[i][j][k] / 255
    return normalizedImage


def rgbToHsv(image, binNumber):
    hsvImage = np.zeros((96, 96, 3), float)
    image = l1NormalizationImage(image)

    for i in range(96):
        for j in range(96):
            H, S, V = 0, 0, 0
            R, G, B = float(image[i][j][0]), float(image[i][j][1]), float(image[i][j][2])
            c_max = max(image[i][j])
            c_min = min(image[i][j])
            deltaC = c_max - c_min

            if deltaC == 0:
                H = 0
            elif c_max == R:
                H = (((G-B) / deltaC) % 6) / 6
            elif c_max == G:
                H = (((B-R) / deltaC) + 2) / 6
            elif c_max == B:
                H = (((R-G) / deltaC) + 4) / 6
            V = c_max

            if V == 0:
                S = 0
            elif V > 0:
                S = deltaC / c_max

            hsvImage[i][j][0] = H*255
            hsvImage[i][j][1] = S*255
            hsvImage[i][j][2] = V*255

    return hsvImage


def perChannelHistogram(image, interval, binNumber):
    # image.size = 96 96 3 pixel x pixel x rgb
    imageX = image.shape[0]
    imageY = image.shape[1]
    histogram = np.zeros([3, binNumber], float)
    for x in range(imageX):
        for y in range(imageY):
            pixelR = int(
                image[x][y][0] // interval)
            pixelG = int(image[x][y][1] // interval)
            pixelB = int(image[x][y][2] // interval)
            histogram[0][pixelR] += 1  # R++
            histogram[1][pixelG] += 1  # G++
            histogram[2][pixelB] += 1  # B++
    return histogram


def threeDHistogram(image, interval, binNumber):
    imageX = image.shape[0]
    imageY = image.shape[1]
    histogram = np.zeros((binNumber, binNumber, binNumber), float)
    for x in range(imageX):
        for y in range(imageY):
            pixelR = int(
                image[x][y][0] // interval)
            pixelG = int(image[x][y][1] // interval)
            pixelB = int(image[x][y][2] // interval)
            histogram[pixelR][pixelG][pixelB] += 1  # R G B ++
    return histogram


def pcAccuracy(imageH, supportH, binNumber):
    R, G, B = 0, 0, 0
    for b in range(binNumber):
        R += min(imageH[0][b], supportH[0][b])

    for b in range(binNumber):
        G += min(imageH[1][b], supportH[1][b])

    for b in range(binNumber):
        B += min(imageH[2][b], supportH[2][b])

    return (R + G + B) / 3


def threeDAccuracy(imageH, supportH, binNumber):
    accuracy = 0
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(binNumber):
                accuracy += min(imageH[i][j][k], supportH[i][j][k])
    return accuracy


def main(histogramType, binNumber, grid, query, isHSV):
    jpeg_names = open("InstanceNames.txt").read().splitlines()
    jpegsSize = len(jpeg_names)
    jpegs = []
    supports = []
    imageHistograms = []
    supportHistograms = []

    for i in range(jpegsSize):
        image = cv2.imread("query_{}/{}".format(query, jpeg_names[i]))
        support = cv2.imread("support_96/{}".format(jpeg_names[i]))
        if isHSV:
            image = rgbToHsv(image, binNumber)
            support = rgbToHsv(support, binNumber)
        jpegs.append(image)
        supports.append(support)

    if histogramType == "perChannel":
        if grid == 1:
            interval = 256 / binNumber
            accuracy = 0
            for i in range(jpegsSize):
                image = jpegs[i]
                imageHistogram = perChannelHistogram(image, interval, binNumber)
                l1NormalizationPcHistogram(imageHistogram[0])
                l1NormalizationPcHistogram(imageHistogram[1])
                l1NormalizationPcHistogram(imageHistogram[2])
                imageHistograms.append(imageHistogram)

                supportImage = supports[i]
                supportImageHistogram = perChannelHistogram(supportImage, interval, binNumber)
                l1NormalizationPcHistogram(supportImageHistogram[0])
                l1NormalizationPcHistogram(supportImageHistogram[1])
                l1NormalizationPcHistogram(supportImageHistogram[2])
                supportHistograms.append(supportImageHistogram)

            for i in range(jpegsSize):
                tempSupportHistogram = supportHistograms[i]
                imageHistogram = imageHistograms[i]
                max_accuracy = pcAccuracy(imageHistogram, tempSupportHistogram, binNumber)
                res = 1
                for j in range(jpegsSize):
                    if i == j:
                        continue
                    supportImageHistogram = supportHistograms[j]
                    temp = pcAccuracy(imageHistogram, supportImageHistogram, binNumber)
                    if temp > max_accuracy:
                        res = 0
                        break

                accuracy += res

            print(str(accuracy/2) + "%")

        else:
            interval = 256 / binNumber
            jpegsSize = len(jpeg_names)
            gSize = int(np.sqrt(grid))
            imageHistograms = []
            supportHistograms = []
            for i in range(jpegsSize):
                image = jpegs[i]
                supportImage = supports[i]
                supportHistogramsList = []
                imageHistogramsList = []
                for j in range(gSize):
                    for k in range(gSize):
                        lx = (j * image.shape[0]) // gSize
                        rx = ((j + 1) * image.shape[0]) // gSize
                        ly = (k * image.shape[1]) // gSize
                        ry = ((k + 1) * image.shape[1]) // gSize

                        tempSupportHistogram = perChannelHistogram(supportImage[lx:rx, ly:ry], interval, binNumber)
                        tempImageHistogram = perChannelHistogram(image[lx:rx, ly:ry], interval, binNumber)
                        supportHistogramsList.append(tempSupportHistogram)
                        imageHistogramsList.append(tempImageHistogram)
                imageHistograms.append(imageHistogramsList)
                supportHistograms.append(supportHistogramsList)
            accuracy = 0
            i = 0
            for i in range(jpegsSize):
                for g in range(grid):
                    imageHistograms[i][g][0] = l1NormalizationPcHistogram(imageHistograms[i][g][0])
                    imageHistograms[i][g][1] = l1NormalizationPcHistogram(imageHistograms[i][g][1])
                    imageHistograms[i][g][2] = l1NormalizationPcHistogram(imageHistograms[i][g][2])

                    supportHistograms[i][g][0] = l1NormalizationPcHistogram(supportHistograms[i][g][0])
                    supportHistograms[i][g][1] = l1NormalizationPcHistogram(supportHistograms[i][g][1])
                    supportHistograms[i][g][2] = l1NormalizationPcHistogram(supportHistograms[i][g][2])
            for i in range(jpegsSize):
                max_accuracy = 0

                for g2 in range(grid):
                    max_accuracy += pcAccuracy(imageHistograms[i][g2], supportHistograms[i][g2], binNumber)
                max_accuracy /= grid
                res = 1
                for m in range(jpegsSize):
                    if i == m:
                        continue
                    temp = 0
                    for gg in range(grid):
                        temp += pcAccuracy(imageHistograms[i][gg], supportHistograms[m][gg], binNumber)
                    temp /= grid

                    if temp > max_accuracy:
                        res = 0
                        break
                accuracy += res
            print(str(accuracy / 2) + "%")

    elif histogramType == "3D":
        if grid == 1:
            interval = 256 / binNumber
            jpegsSize = len(jpeg_names)

            for i in range(jpegsSize):
                image = jpegs[i]
                imageHistogram = threeDHistogram(image, interval, binNumber)
                l1Normalization3DHistogram(imageHistogram)
                imageHistograms.append(imageHistogram)
                supportImage = supports[i]
                supportImageHistogram = threeDHistogram(supportImage, interval, binNumber)
                l1Normalization3DHistogram(supportImageHistogram)
                supportHistograms.append(supportImageHistogram)

            accuracy = 0
            for i in range(jpegsSize):
                imageHistogram = imageHistograms[i]
                tempSupportImageHistogram = supportHistograms[i]
                max_accuracy = threeDAccuracy(imageHistogram, tempSupportImageHistogram, binNumber)
                res = 1
                for j in range(jpegsSize):
                    if i == j:
                        continue
                    temp = threeDAccuracy(imageHistogram, supportHistograms[j], binNumber)

                    if temp > max_accuracy:
                        res = 0
                        break
                accuracy += res

            print(str(accuracy / 2) + "%")
        else:
            interval = 256 / binNumber
            jpegsSize = len(jpeg_names)
            gSize = int(np.sqrt(grid))
            for i in range(jpegsSize):
                image = jpegs[i]
                supportImage = supports[i]
                supportHistogramsList = []
                imageHistogramsList = []
                for j in range(gSize):
                    for k in range(gSize):
                        lx = (j * image.shape[0]) // gSize
                        rx = ((j + 1) * image.shape[0]) // gSize
                        ly = (k * image.shape[1]) // gSize
                        ry = ((k + 1) * image.shape[1]) // gSize
                        tempSupportHistogram = threeDHistogram(supportImage[lx:rx, ly:ry], interval, binNumber)
                        tempImageHistogram = threeDHistogram(image[lx:rx, ly:ry], interval, binNumber)
                        supportHistogramsList.append(tempSupportHistogram)
                        imageHistogramsList.append(tempImageHistogram)
                imageHistograms.append(imageHistogramsList)
                supportHistograms.append(supportHistogramsList)
            accuracy = 0

            for i in range(jpegsSize):
                for g in range(grid):
                    imageHistograms[i][g] = l1Normalization3DHistogramGrid(imageHistograms[i][g], grid)
                    supportHistograms[i][g] = l1Normalization3DHistogramGrid(supportHistograms[i][g], grid)

            for i in range(jpegsSize):
                max_accuracy = 0
                for g2 in range(grid):
                    max_accuracy += threeDAccuracy(imageHistograms[i][g2], supportHistograms[i][g2], binNumber)
                # accuracy hesaplandı aynıları arasında
                max_accuracy /= grid
                res = 1
                for m in range(jpegsSize):
                    if i == m:
                        continue
                    temp = 0
                    for gg in range(grid):
                        temp += threeDAccuracy(imageHistograms[i][gg], supportHistograms[m][gg], binNumber)
                    temp /= grid

                    if temp > max_accuracy:
                        res = 0
                        break
                accuracy += res

            print(str(accuracy/2) + "%")
    return 0

