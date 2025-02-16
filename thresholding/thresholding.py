"""
    Thresholding Viewer.

    Copyright (c) 2025 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import cv2
import numpy as np
import cvui

WINDOW_NAME = "Binarize Viewer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")

    args = parser.parse_args()

    cvui.init(WINDOW_NAME)
    gaussian_block_size = [11.0]
    gaussian_c_value = [2]

    mean_block_size = [11.0]
    mean_c_value = [2]

    threshold_value = [127]

    niblack_block_size_value = [51.0]
    niblack_k_value = [-0.2]

    sauvola_block_size_value = [51.0]
    sauvola_k_value = [0.5]
    sauvola_r_value = [128]

    wolf_block_size_value = [51.0]
    wolf_k_value = [-0.2]

    nick_block_size_value = [51.0]
    nick_k_value = [-0.2]

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    while True:

        # Adaptive thresholding with Gaussian filter
        adaptive_gaussian_binary_image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(gaussian_block_size[0]),
            int(gaussian_c_value[0]),
        )

        # Adaptive thresholding with mean filter
        adaptive_mean_binary_image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            int(mean_block_size[0]),
            int(mean_c_value[0]),
        )

        # Otsu's thresholding
        _, otsu_binary_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Thresh Triangle
        _, thresh_triangle_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )

        # Thresholding
        _, thresh_image = cv2.threshold(
            image, threshold_value[0], 255, cv2.THRESH_BINARY
        )

        # Niblack thresholding
        niblack_binary_image = cv2.ximgproc.niBlackThreshold(
            image,
            maxValue=255,
            type=cv2.THRESH_BINARY,
            blockSize=int(niblack_block_size_value[0]),
            k=niblack_k_value[0],
            binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK,
        )

        # Sauvola thresholding
        sauvola_binary_image = cv2.ximgproc.niBlackThreshold(
            image,
            maxValue=255,
            type=cv2.THRESH_BINARY,
            blockSize=int(sauvola_block_size_value[0]),
            k=sauvola_k_value[0],
            r=sauvola_r_value[0],
            binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
        )

        # Wolf thresholding
        wolf_binary_image = cv2.ximgproc.niBlackThreshold(
            image,
            maxValue=255,
            type=cv2.THRESH_BINARY,
            blockSize=int(wolf_block_size_value[0]),
            k=wolf_k_value[0],
            binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF,
        )

        # NICK thresholding
        nick_binary_image = cv2.ximgproc.niBlackThreshold(
            image,
            maxValue=255,
            type=cv2.THRESH_BINARY,
            blockSize=int(nick_block_size_value[0]),
            k=nick_k_value[0],
            binarizationMethod=cv2.ximgproc.BINARIZATION_NICK,
        )

        display_image = np.full((h * 4, w * 4), 127, np.uint8)

        display_image[0:h, 0:w] = image
        display_image[0:h, w : w * 2] = otsu_binary_image
        display_image[0:h, w * 2 : w * 3] = adaptive_gaussian_binary_image

        display_image[h : h * 2, 0:w] = adaptive_mean_binary_image
        display_image[h : h * 2, w : w * 2] = thresh_triangle_image
        display_image[h : h * 2, w * 2 : w * 3] = thresh_image

        display_image[h * 2 : h * 3, 0:w] = niblack_binary_image
        display_image[h * 2 : h * 3, w : w * 2] = sauvola_binary_image
        display_image[h * 2 : h * 3, w * 2 : w * 3] = wolf_binary_image

        display_image[h * 3 : h * 4, 0:w] = nick_binary_image

        x = w * 3 + 10
        width = w - 20

        # display
        cvui.text(display_image, 5, 5, "Original", 0.6, 0)
        cvui.text(display_image, w + 5, 5, "Out", 0.6, 0)
        cvui.text(display_image, w * 2 + 5, 5, "Adaptive thresh (Gaussian)", 0.6, 0)

        cvui.text(display_image, 5, h + 5, "Adaptive thresh (Mean)", 0.6, 0)
        cvui.text(display_image, w + 5, h + 5, "Thresh Triangle", 0.6, 0)
        cvui.text(display_image, w * 2 + 5, h + 5, "Thresholding", 0.6, 0)

        cvui.text(display_image, 5, h * 2 + 5, "Niblack's technique", 0.6, 0)
        cvui.text(display_image, w + 5, h * 2 + 5, "Sauvola's technique", 0.6, 0)
        cvui.text(display_image, w * 2 + 5, h * 2 + 5, "Wolf's technique", 0.6, 0)

        cvui.text(display_image, 5, h * 3 + 5, "NICK technique", 0.6, 0)

        # Adaptive thresholding with Gaussian filter
        # block size
        cvui.rect(display_image, x - 5, 5, width + 5, 150, 255)
        cvui.text(display_image, x, 10, "Adaptive thresholding (Gaussian)")
        cvui.text(display_image, x, 25, "Block Size")
        cvui.trackbar(
            display_image,
            x + 5,
            40,
            width,
            gaussian_block_size,
            3.0,
            50.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 80, "C")
        cvui.trackbar(
            display_image,
            x,
            95,
            width,
            gaussian_c_value,
            2.0,
            50.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            1.0,
        )

        # Adaptive thresholding with Mean filter
        cvui.rect(display_image, x - 5, 170, width + 5, 150, 255)
        cvui.text(display_image, x, 175, "Adaptive thresholding (Mean)")
        cvui.text(display_image, x, 190, "Block Size")
        cvui.trackbar(
            display_image,
            x,
            210,
            width,
            mean_block_size,
            3.0,
            50.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 250, "C")
        cvui.trackbar(
            display_image,
            x,
            265,
            width,
            mean_c_value,
            2.0,
            50.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            1.0,
        )

        # Thresholding
        cvui.rect(display_image, x - 5, 335, width + 5, 80, 255)
        cvui.text(display_image, x, 340, "Thresholding")
        cvui.trackbar(
            display_image,
            x,
            360,
            width,
            threshold_value,
            0.0,
            255.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            1.0,
        )

        # Classic Niblack binarization
        cvui.rect(display_image, x - 5, 430, width + 5, 150, 255)
        cvui.text(display_image, x, 435, "Classic Niblack binarization")
        cvui.text(display_image, x, 450, "Block Size")
        cvui.trackbar(
            display_image,
            x,
            470,
            width,
            niblack_block_size_value,
            3.0,
            100.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 510, "K")
        cvui.trackbar(
            display_image,
            x,
            530,
            width,
            niblack_k_value,
            -1.0,
            1.0,
            0.0,
            "%.1Lf",
        )

        # Sauvola's technique
        cvui.rect(display_image, x - 5, 595, width + 5, 210, 255)
        cvui.text(display_image, x, 600, "Sauvola's technique")
        cvui.text(display_image, x, 615, "Block Size")
        cvui.trackbar(
            display_image,
            x,
            635,
            width,
            sauvola_block_size_value,
            3.0,
            100.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 675, "K")
        cvui.trackbar(
            display_image,
            x,
            690,
            width,
            sauvola_k_value,
            -1.0,
            1.0,
            0.0,
            "%.1Lf",
        )
        cvui.text(display_image, x, 730, "R")
        cvui.trackbar(
            display_image,
            x,
            750,
            width,
            sauvola_r_value,
            0.0,
            255.0,
            0.0,
            "%.1Lf",
        )

        # Wolf's technique
        cvui.rect(display_image, x - 5, 820, width + 5, 150, 255)
        cvui.text(display_image, x, 825, "Wolf's technique")
        cvui.text(display_image, x, 840, "Block Size")
        cvui.trackbar(
            display_image,
            x,
            860,
            width,
            wolf_block_size_value,
            3.0,
            100.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 900, "K")
        cvui.trackbar(
            display_image,
            x,
            920,
            width,
            wolf_k_value,
            -1.0,
            1.0,
            0.0,
            "%.1Lf",
        )

        # NICK's technique
        cvui.rect(display_image, x - 5, 985, width + 5, 150, 255)
        cvui.text(display_image, x, 990, "NICK's technique")
        cvui.text(display_image, x, 1005, "Block Size")
        cvui.trackbar(
            display_image,
            x,
            1015,
            width,
            nick_block_size_value,
            3.0,
            100.0,
            1.0,
            "%.1Lf",
            cvui.TRACKBAR_DISCRETE,
            2.0,
        )
        cvui.text(display_image, x, 1045, "K")
        cvui.trackbar(
            display_image,
            x,
            1055,
            width,
            nick_k_value,
            -1.0,
            1.0,
            0.0,
            "%.1Lf",
        )

        cvui.update()

        cv2.imshow(WINDOW_NAME, display_image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
