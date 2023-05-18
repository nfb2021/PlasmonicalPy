import cv2 as cv
from cv2 import dnn_superres
import os
import numpy as np
from scipy.signal import fftconvolve, savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from skimage.measure import profile_line
from typing import Optional, List, Tuple
from shapely.geometry import LineString, Point
from descartes import PolygonPatch

###################################################################################################################################################################
###################################################################################################################################################################

plt.rc("font", size=25)
figsize = (11, 8)
fontsize = 25
fontsize_legend = 15
markersize = 18
dpi = 200
cmap = mpl.cm.get_cmap("turbo").copy()
marker_letters = [
    "$a$",
    "$b$",
    "$c$",
    "$d$",
    "$e$",
    "$f$",
    "$g$",
    "$h$",
    "$i$",
    "$j$",
    "$k$",
    "$l$",
    "$m$",
    "$n$",
    "$o$",
    "$p$",
    "$q$",
    "$r$",
    "$s$",
    "$t$",
    "$u$",
    "$v$",
    "$w$",
    "$x$",
    "$y$",
    "$z$",
]

###################################################################################################################################################################
###################################################################################################################################################################


class CsvToImg:
    """Creates a corrected BFP image of specified format. Not required for main functionalities, just add-on.

    :param csv_file: Name of the csv file containing the non-corrected BFP image
    :type csv_file: str
    :param csv_reference: Name of the csv file containing the reference BFP image
    :type csv_reference: str
    :param out_format: Format ouf image to be written (png, jpg, pdf, tif)
    :type out_format: Optional[str]
    :param show_init: If the image should be displayed
    :type show_init: Optional[bool]
    """

    def __init__(
        self,
        csv_file: str,
        csv_reference: str,
        out_format: Optional[str] = "png",
        show_init: Optional[bool] = False,
    ) -> None:
        self.csv_file: str = csv_file
        self.filename: str = csv_file.split(".")[0]
        self.out_format: str = out_format

        # try to find specified csv file, otherwise abort
        try:
            with open(self.csv_file, "r") as f:
                self.raw: np.ndarray = np.loadtxt(f, delimiter=" ", dtype=float)

            if show_init == True:
                fig = plt.figure(1, figsize=figsize)
                ax0 = fig.addsubplot(131)
                ax0.set_title("Image")
                ax0.imshow(self.raw)

        except FileNotFoundError:
            print(
                f'The specified file "{self.csv_file}" was not found in the current workign directory. Program is terminated.'
            )
            exit()

        # try to find specified reference csv file, otherwise program continues without correction
        try:
            with open(csv_reference, "r") as f:
                self.ref: np.ndarray = np.loadtxt(f, delimiter=" ", dtype=float)

            if show_init == True:
                ax1 = fig.add_subplot(132)
                ax1.set_title("Reference")
                ax1.imshow(self.ref)

            # correction
            self.arr: np.ndarray = self.raw - self.ref

            if show_init == True:
                ax2 = fig.add_subplot(133)
                ax2.set_title("Image - Reference")
                ax2.imshow(self.arr)

                plt.show()

        except FileNotFoundError:
            print(
                f'The specified reference "{csv_reference}" was not found in the current workign directory. Continuing without reference.'
            )

            # continue without correction
            self.arr = self.raw

        self.img_uint8: np.ndarray = np.array(self.arr).astype(
            "uint8"
        )  # 8-bit grayscale image
        self.img_float: np.ndarray = np.array(self.arr).astype(
            "float64"
        )  # grayscale image of float type, for further manipulations

        self.uint8_name: str = f"{self.filename}_original.{self.out_format}"

        cv.imwrite(self.uint8_name, self.img_uint8)  # writes image and creates file


class Image:
    """Toolbox for basic image manipulation.
    Prefers csv files over image files, meaning: The constructor deduces the name of the original csv file and csv reference, based on specified input image name.
    If these csv files exist, the constructor loads their data. Otherwise, the image file itself is taken as source of data.

    :param img: Name of the image to be loaded. Name must correspond to name of original csv file
    :type img: str
    """

    def __init__(self, img: str):
        if os.path.isfile(img):
            self.root: str = os.path.abspath(img)
            self.name: str = img.split(r"\\")[-1].split(".")[0]

            self.corresponding_csv: str = str(img.split("_original")[0]) + f".csv"

            if os.path.isfile(self.corresponding_csv):
                meas_split = self.corresponding_csv.split("_")
                meas_idx = [idx for idx, x in enumerate(meas_split) if "Meas" in x][0]
                meas_split.pop(meas_idx)

                refs = [
                    x
                    for x in os.listdir()
                    if x.endswith(".csv") and "spectrum" not in x and "Ref" in x
                ]

                for ref in refs:
                    ref_split = ref.split("_")
                    ref_split.pop(meas_idx)
                    if meas_split == ref_split:
                        _meas_num = self.corresponding_csv.split("_")[meas_idx]
                        _ref_num = ref.split("_")[meas_idx]

                        meas_num = float(
                            ("").join([x for x in _meas_num if x in "0123456789"])
                        )
                        ref_num = float(
                            ("").join([x for x in _ref_num if x in "0123456789"])
                        )

                        if meas_num == ref_num:
                            self.csv_meas = self.corresponding_csv
                            self.csv_ref = ref

                            try:
                                with open(self.csv_meas, "r") as f:
                                    raw = np.loadtxt(f, delimiter=" ", dtype=float)

                            except FileNotFoundError:
                                print(
                                    f'The specified file "{self.csv_meas}" was not found in the current working directory. Program is terminated.'
                                )
                                exit()

                            try:
                                with open(self.csv_ref, "r") as f:
                                    try:
                                        ref = np.loadtxt(f, delimiter=" ", dtype=float)
                                    except ValueError:
                                        ref = np.loadtxt(f, delimiter=",", dtype=float)

                                try:
                                    arr = raw - ref
                                except ValueError:
                                    ref = np.concatenate(
                                        (ref, [np.zeros(len(ref[0]))]), axis=0
                                    )
                                    arr = raw - ref

                            except FileNotFoundError:
                                print(
                                    f'The specified reference "{self.csv_ref}" was not found in the current workign directory. Continuing without reference.'
                                )

                                arr = raw

            else:
                print(
                    f"\n\t\tThe csv-file corresponding to the specified image ({self.corresponding_csv}, {img}) was not found in the current working directory. Therefore, the image itself will be read in as source of data."
                )
                arr = cv.imread(img)
                try:
                    arr = cv.cvtColor(arr.cv.COLOR_BGR2GRAY)
                except:
                    pass

            self.img_non_uint8: np.ndarray = arr
            self.img: np.ndarray = np.array(arr).astype("uint8")

            self.img_copy: np.ndarray = np.copy(self.img)

            try:
                self.gray: np.ndarray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            except:
                self.gray: np.ndarray = self.img

            # self.color = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
            self.blur: np.ndarray = cv.GaussianBlur(
                self.gray, (5, 5), cv.BORDER_DEFAULT
            )  # blurred version obtained by convolution with 5x5 Gaussian kernel
            _, self.binary_otsu = cv.threshold(
                self.blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )  # binary obtaiend by Otsu's method
            _, self.binary_median = cv.threshold(
                self.blur, ndimage.median(self.blur), 255, cv.THRESH_BINARY
            )  # binary obtaiend by taking median as threshold
            self.width, self.length = self.gray.shape[:]  # image dimensions

        else:
            print(f'\nThe specified file "{img}" was not found.')
            exit()

    def rotate_180_deg(self, img: np.ndarray) -> np.ndarray:
        """Rotates image by 180deg

        :param img: The image
        :type img: np.ndarray
        :returns: Rotated image
        :rtype: np.ndarray
        """

        return cv.flip(img, -1)  # flipped over x- and y-axis == rotated 180 deg

    def get_fourier_transform(self, img: np.ndarray) -> np.ndarray:
        """Centered Fourier transformation of image

        :param img: The image
        :type img: np.ndarray
        :returns: Centered Fourier transform
        :rtype: np.ndarray
        """

        return np.fft.fftshift(np.fft.fft2(img))

    def get_inverse_fourier_transform(self, ft_img: np.ndarray) -> np.ndarray:
        """Inverse Fourier transformation

        :param ft_img: The Fourier transform of an image
        :type ft_img: np.ndarray
        :returns: Inverse Fourier transform, a real space image
        :rtype: np.ndarray
        """

        return abs(np.fft.ifft2(ft_img))

    def get_flipped(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flips image over horizontal and vertical

        :param img: The image
        :type img: np.ndarray
        :returns: Version of image flipped over x-axis and image flipped over y-axis
        :rtype: np.ndarray, np.ndarray
        """

        return cv.flip(img, 0), cv.flip(img, 1)  # flipped over x (0) and y (1) axis

    def normalize_for_cbar(self, img: np.ndarray) -> np.ndarray:
        """Normalizes an image to pixel values [0, 1]

        :param img: The image
        :type img: np.ndarray
        :returns: Normalized version of image
        :rtype: np.ndarray
        """

        img_copy = np.copy(img)
        img_copy = np.array(img_copy).astype("float64")
        # img_copy -= np.amin(img_copy)
        img_copy /= np.amax(img_copy)

        return img_copy

    def get_contours(
        self, binary: np.ndarray, show: Optional[bool] = True
    ) -> Tuple[np.ndarray, float]:
        """Contour detection algorithm

        :param binary: The image as binary version
        :type binary: np.ndarray
        :param show: If image with highlighted identified contours should be displayed
        :type show: Optional[bool]
        :returns: Contours and hierarchies
        :rtype: np.ndarray, float
        """

        contours, hierarchies = cv.findContours(
            binary, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
        )
        if show == True:
            cv.drawContours(self.img_copy, contours, -1, (0, 0, 255), 1)
            cv.imshow("In FCA: Contours", self.img_copy)
            cv.waitKey(0)

        return contours, hierarchies

    def get_longest_contour(self, contours: np.ndarray) -> List[np.ndarray]:
        """Returns longest contour out of list of multiple contours (e.g. return value of method 'get_contours'). Longest contour corresponds to structure of interest usually, shorter contours to noise/...

        :param contours: Previously identified contours
        :type contours: np.ndarray
        :returns: List containing longest contours as only element
        :rtype: List[np.ndarray]
        """

        _longest_contour = 0
        print(len(contours))
        for c, contour in enumerate(contours):
            if len(contour) > _longest_contour:
                _longest_contour = contour

        return [_longest_contour]

    def get_coords_from_contours(
        self, contour_list: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Extracts x and y coordinates from provided list of contours (e.g. return value of method 'get_longest_contour')

        :param contour_list: Previously identified contours
        :type contour_list: List or np.ndarray
        :returns: List of x coordinates and list of corresponding y coordinates
        :rtype: List, List
        """

        xx, yy = [], []
        contours = contour_list[
            0
        ]  # only takes one found contour into account, thats why the longest contour is stored in a list as single element
        for px in contours:
            xx.append(px[0][0])
            yy.append(px[0][1])

        return xx, yy

    def mask_background(
        self,
        img_gray: np.ndarray,
        lower_thres: Optional[int] = 12,
        upper_thres: Optional[int] = 242,
        pad: Optional[float] = 0.1,
        show: Optional[bool] = False,
        save: Optional[bool] = True,
        show_hist: Optional[bool] = False,
    ) -> Tuple[np.ndarray, str]:
        """Takes an input gray image and masks the background, returning the masked image. This is done in two steps:
        1. All pixel values smaller than the lower and larger than the upper threshold are set to zero, using a histogram. This step increases the chance of the contour of the BFP being identified correctly.
        2. Using the identified BFP, this part of the image is cropped and put atop of a black canvas (all pixel values equal zero). The space around the BFP crop out and the border of the black canvas are set via the pad paramater.
        The combination of both steps ensures a uniform black background, up to a certain degeree indepednent of the S/N ratio. Nonetheless, pixel contained inside the BFP might be set to zero, too.

        :param img_gray: gray scale 8-bit image
        :type img_gray: np.ndarray
        :param lower_thres: Specifies the lower threshold
        :type lower_thres: Optional[int]
        :param upper_thres: Specifies the upper threshold
        :type upper_thres: Optional[int]
        :param pad: In percent relative to longest main axis through BFP crop put. The pad is added on both sides of the crop out, setting the size of the resulting square black canvas and function return
        :type pad: Optional[float]
        :param show: If the image should be displayed
        :type show: Optional[bool]
        :param save: If the image should be saved
        :type save: Optional[bool]
        :param show_hist: If the histogram used in the first step as well as the thresholds should be displayed. Via user prompt, the thresholds can be set manually then and the thresholds set during the function call will be overriden
        :type show_hist: Optional[bool]
        :returns: Masked image (= BFP crop out ontop of black canvas, 8-bit square image) and filename of saved image
        :rtype: np.ndarray, str
        """

        gray = np.copy(img_gray).astype("uint8")
        # lower_thres, upper_thres = int(thres_factor * np.amax(gray)), int((1 - thres_factor) * np.amax(gray))

        if show_hist == True:
            fig = plt.figure(figsize=(14, 8))
            ax0 = fig.add_subplot(111)
            ax0.set_title("Histogram of pixel values")
            histg = cv.calcHist([gray], [0], None, [256], [0, 256])
            histg_x = np.linspace(0, 255, 256)
            ax0.plot(histg_x, histg, color="black", label="histogram")
            ax0.axvline(
                lower_thres, color="red", label=f"lower threshold: {lower_thres}"
            )
            ax0.axvline(
                upper_thres, color="blue", label=f"upper threshold: {upper_thres}"
            )
            ax0.set_xlabel("Pixel Value")
            ax0.set_ylabel("Counts")
            plt.legend()

            print(
                f"\nIn the next step you can adapt the thresholds, after you close the histogram figure. Both thresholds are to be specified as pixel values. Be careful not to exceed the maximum pixel values {np.amin(gray), np.amax(gray)}."
            )
            plt.show()

            choice = input(
                '\nDo wou want to adapt the thresholds by entering new values in the form of "lower threshold, upper threshold", as their respective pixel value? This action will override the specified thresholds, before the program continues. If the thresholds already are as desired, enter "n" and the program continues. Confirm your input by pressing the ENTER.'
            )

            if choice == "n":
                print("The thresholds are not overriden")

            elif "," in choice:
                lower_thres, upper_thres = int(choice.split(",")[0]), int(
                    choice.split(",")[1]
                )
                if np.amin(
                    gray
                ) <= lower_thres < upper_thres and upper_thres <= np.amax(gray):
                    print(
                        f"\nThe lower threshold was overriden to a pixel value of {lower_thres}, the upper threshold to {upper_thres}. Program continues."
                    )
                else:
                    print("Invalid input. Program abort.")
                    exit()

            else:
                print("Invalid input. Program abort.")
                exit()

            plt.close()

        for r, row in enumerate(gray):
            for p, px in enumerate(row):
                if px <= lower_thres or px >= upper_thres:
                    gray[r][p] = 0

        blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        blur = np.array(blur).astype("uint8")
        _, binary_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = self.get_contours(binary_otsu, show=False)

        if len(contours) > 1:
            contours = self.get_longest_contour(contours)

        xx, yy = self.get_coords_from_contours(contours)
        _x_diff, _y_diff = np.abs(np.max(xx) - np.min(xx)), np.abs(
            np.max(yy) - np.min(yy)
        )

        if _x_diff >= _y_diff:
            background_length = _x_diff
        else:
            background_length = _y_diff

        masked_background = np.zeros(
            [
                int((1 + 2 * pad) * background_length),
                int((1 + 2 * pad) * background_length),
            ]
        )

        for y in range(len(gray)):
            for x in range(len(gray[0])):
                if np.min(xx) <= x <= np.max(xx) and np.min(yy) <= y <= np.max(yy):
                    masked_background[y - np.min(yy) + int(pad * background_length)][
                        x - np.min(xx) + int(pad * background_length)
                    ] = gray[y][x]

        masked_background = np.array(masked_background).astype("uint8")

        if show == True:
            fig = plt.figure(2020)
            ax = fig.add_subplot(111)
            _masked_background = self.normalize_for_cbar(masked_background)
            masked_array = np.ma.array(
                _masked_background, mask=(_masked_background == 0)
            )
            im = ax.imshow(masked_array, cmap=cmap)
            plt.axis("off")
            cmap.set_bad("black", 1.0)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical", label="Intensity [a.u.]")

            plt.show()

        if save == True:
            out_name = f"{self.name}_masked.png"
            cv.imwrite(out_name, masked_background)

        else:
            out_name = None

        return masked_background, out_name

    def get_super_res(
        self,
        img_gray: np.ndarray,
        model: Optional[str] = "EDSR_x3.pb",
        blur: Optional[bool] = False,
        show: Optional[bool] = False,
        save: Optional[bool] = True,
    ) -> Tuple[np.ndarray, str]:
        """Upscaling of grayscale image to superresolution image, using the specified machine learning model. This function is based on https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066. Refer to this website for further information.

        :param model: Machine learning model to be used
        :type model: str
        :param blur: If resulting image should be blurred using a 3x3 Gaussian kernel
        :type blur: Optional[bool]
        :param show: If resulting he image should be displayed
        :type show: Optional[bool]
        :param save: If resulting image should be saved
        :type save: Optional[bool]
        :returns: Upscaled superresolution 8-bit image and filename of saved image
        :rtype: np.ndarray, str
        """

        try:
            img_colored = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

        except:
            img_colored = img_gray

        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()
        # Read the desired model
        try:
            sr.readModel(os.path.join("opencv_superres_models", model))
        except:
            sr.readModel(os.path.join("..", "..", "opencv_superres_models", model))

        # Set the desired model and scale to get correct pre- and post-processing
        name = model.split("_")[0].lower()
        num = int(model.split(".")[0][-1])
        sr.setModel(name, num)

        # Upscale the image
        print("\t\tUpscaling image...")
        img_gray_superres = cv.cvtColor(sr.upsample(img_colored), cv.COLOR_BGR2GRAY)
        print("\t\tUpscaling succesfull\n")

        if blur == True:
            img_gray_superres = cv.GaussianBlur(
                img_gray_superres, (3, 3), cv.BORDER_DEFAULT
            )

        if show == True:
            fig = plt.figure(2021)
            ax = fig.add_subplot(111)
            _img_gray_superres = self.normalize_for_cbar(img_gray_superres)
            masked_array = np.ma.array(
                _img_gray_superres, mask=(_img_gray_superres == 0)
            )
            im = ax.imshow(masked_array, cmap=cmap)
            plt.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_title("Superresolution Image")
            fig.colorbar(im, cax=cax, orientation="vertical", label="Intensity [a.u.]")
            plt.show()

        if save == True:
            model_name = model.split(".")[0]
            out_name = f"{self.name}_{model_name}_superres.png"
            cv.imwrite(out_name, img_gray_superres)

        else:
            out_name = None

        return img_gray_superres, out_name

    def make_false_color_img(self, img_gray: np.ndarray, img_name: str) -> None:
        """Creates false color image, as pdf and as same format of input image

        :param img_gray: Gray scale image
        :type img_gray: np.ndarray
        :param img_name: Name of image. Used to create name for output image
        :type img_name: str
        """

        fig = plt.figure(2020, figsize=(7, 6))
        ax = fig.add_subplot(111)
        _masked_background = self.normalize_for_cbar(img_gray)
        # _masked_background = img_gray
        masked_array = np.ma.array(_masked_background, mask=(_masked_background == 0))
        im = ax.imshow(masked_array, cmap=cmap)
        plt.axis("off")
        cmap.set_bad("black", 1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical", label="Intensity [a.u.]")
        plt.tight_layout()
        plt.subplots_adjust(left=+0.05, bottom=0.1, right=0.85, top=0.9, wspace=0.2)
        _name = img_name.split(".")[0]
        ending = img_name.split(".")[1]
        plt.savefig(f"{_name}_colored.{ending}", dpi=200)
        plt.savefig(f"{_name}_colored.pdf", dpi=200)
        plt.close()


class FourierCorrelationAnalysis(Image):
    """Algorithm to conduct Fourier correlation analysis

    :param img: Name of the image to be loaded. Name must correspond to name of original csv file
    :type img: str
    """

    def __init__(self, img: str):
        super().__init__(img)

    def get_correlation_coeff(self, img2: np.ndarray) -> np.ndarray:
        """Calculates Fourier correlation coefficient map

        :param img2: Gray scale image
        :type img2: np.ndarray
        :returns: Fourier correlation coefficient map (power spectrum)
        :rtype: np.ndarray
        """

        img1: np.ndarray = self.gray.astype("float")
        img2 = img2.astype("float")

        # get rid of the averages, otherwise the results are not good
        img1 -= np.mean(img1)
        img2 -= np.mean(img2)

        # calculate the correlation image; note the flipping of onw of the images
        img2 = self.rotate_180_deg(img2)

        return fftconvolve(img1, img2, mode="same")

    def get_correlation_coeff_alternative(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> np.ndarray:
        """Alternative: Calculates Fourier correlation coefficient map

        :param img1: Gray scale image
        :type img1: np.ndarray
        :param img2: Gray scale image
        :type img2: np.ndarray
        :returns: Fourier correlation coefficient map (power spectrum)
        :rtype: np.ndarray
        """

        ft1 = self.get_fourier_transform(img1)
        ft2 = self.get_fourier_transform(img2)

        cross_power_spectrum = (ft1 * ft2.conj()) / np.abs(ft1 * ft2.conj())
        r = self.get_inverse_fourier_transform(cross_power_spectrum)

        return r

    def get_max_coords(self, img: np.ndarray) -> list[int, int]:
        """Coordinates of maximum pixel in an image

        :param img: Gray scale image
        :type img: np.ndarray
        :returns: Coordinates
        :rtype: List[int, int]
        """

        index = np.where(img == np.amax(img))
        return list(zip(index[1], index[0]))

    def get_center_coords(self) -> Tuple[float, float]:
        """Heart of the FCA algorithm, as it combines all methods and yields the center coordinates of the structure of interest

        :returns: Center coordinates of the structure of interest
        :rtype: int, int
        """

        h_flip, v_flip = self.get_flipped(self.gray)
        coeff1 = self.get_correlation_coeff(h_flip)
        max_coeff1 = self.get_max_coords(coeff1)

        coeff2 = self.get_correlation_coeff(v_flip)
        max_coeff2 = self.get_max_coords(coeff2)

        if len(max_coeff1) > 1 or len(max_coeff2) > 1:
            print(
                f"\t\nMultiple coordinates with maximum intensity were found. Taking the average\n"
            )

        mc1y = np.sum(list(zip(*max_coeff1))[1]) / len(max_coeff1)
        mc2x = np.sum(list(zip(*max_coeff2))[0]) / len(max_coeff2)

        xc = 0.5 * ((0.5 * self.length) + mc2x)
        yc = 0.5 * ((0.5 * self.width) + mc1y)

        print(f"\n\t\tCenter coordinates: xc: {xc}, yx: {yc}\n")
        return xc, yc

    def plot_fca(self) -> None:
        """Visualization of FCA: original image, both flipped version, self correlation and both flip correlations"""

        h_flip, v_flip = self.get_flipped(self.gray)

        self_corr = self.get_correlation_coeff(self.gray)
        self_corr_max = self.get_max_coords(self_corr)

        coeff_v = self.get_correlation_coeff(v_flip)
        index_v = self.get_max_coords(coeff_v)

        coeff_h = self.get_correlation_coeff(h_flip)
        index_h = self.get_max_coords(coeff_h)

        xc, yc = self.get_center_coords()

        self.gray = self.gray.astype("float64")
        v_flip = v_flip.astype("float64")
        h_flip = h_flip.astype("float64")
        self_corr = self_corr.astype("float64")
        coeff_v = coeff_v.astype("float64")
        coeff_h = coeff_h.astype("float64")

        self.gray -= np.amin(self.gray)
        v_flip -= np.amin(v_flip)
        h_flip -= np.amin(h_flip)
        self_corr -= np.amin(self_corr)
        coeff_v -= np.amin(coeff_v)
        coeff_h -= np.amin(coeff_h)

        self.gray /= np.amax(self.gray)
        v_flip /= np.amax(v_flip)
        h_flip /= np.amax(h_flip)
        self_corr /= np.amax(self_corr)
        coeff_v /= np.amax(coeff_v)
        coeff_h /= np.amax(coeff_h)

        fig = plt.figure(1, figsize=(14, 8))
        fig.suptitle(f"{self.name}")

        ax1 = fig.add_subplot(231)
        ax1.set_title("Gray")
        im1 = ax1.imshow(self.gray, cmap=cmap)
        ax1.plot(xc, yc, "kx", label=f"({xc}, {yc})")

        ax1.legend()
        ax1.tick_params("x", labelbottom=False)

        ax2 = fig.add_subplot(232, sharey=ax1)
        plt.tick_params("y", labelright=False)
        ax2.set_title("Vertically Flipped")
        im2 = ax2.imshow(v_flip, cmap=cmap)

        ax2.tick_params("x", labelbottom=False)
        ax2.tick_params("y", labelleft=False)

        ax3 = fig.add_subplot(233, sharey=ax2)
        ax3.set_title("Horizontally Flipped")
        im3 = ax3.imshow(h_flip, cmap=cmap)

        ax3.tick_params("x", labelbottom=False)
        ax3.tick_params("y", labelleft=False)

        ax4 = fig.add_subplot(234, sharex=ax1)
        ax4.set_title("Selfcorrelation")
        im4 = ax4.imshow(self_corr, cmap=cmap)
        for coord in self_corr_max:
            x, y = coord
            ax4.plot(x, y, "kx", label=f"({x}, {y})")

        ax4.legend()

        ax5 = fig.add_subplot(235, sharey=ax4, sharex=ax2)
        ax5.set_title("Vertical Correlation")
        im5 = ax5.imshow(coeff_v, cmap=cmap)
        for coord in index_v:
            x, y = coord
            plt.plot(x, y, "kx", label=f"({x}, {y})")

        ax5.legend()
        ax5.tick_params("y", labelleft=False)

        ax6 = fig.add_subplot(236, sharey=ax5, sharex=ax3)
        ax6.set_title("Horizontal Correlation")
        im6 = ax6.imshow(coeff_h, cmap=cmap)
        for coord in index_h:
            x, y = coord
            plt.plot(x, y, "kx", label=f"({x}, {y})")

        ax6.legend()
        ax6.tick_params("y", labelleft=False)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        fig.colorbar(im6, cax=cbar_ax)

        plt.tight_layout()

        plt.subplots_adjust(left=+0.05, bottom=0.1, right=0.85, top=0.9, wspace=0.2)
        plt.show()


class Analysis(FourierCorrelationAnalysis):
    """Enables radial intensity distribution analysis.
    An instanciated object of this class has full access to all methods and attributes, since this class is child to all other classes

    :param img: Name of the image to be loaded. Name must correspond to name of original csv file
    :type img: str
    :param out_format: Format of saved image (png, jpeg, tif, pdf)
    :type out_format: Optional[str]
    """

    def __init__(self, img: str, out_format: Optional[str] = "png"):
        super().__init__(img)
        self.out_format: str = out_format

    def get_radial_intensities(
        self,
        only_rad_plot: Optional[bool] = True,
        show: Optional[bool] = True,
        save: Optional[bool] = True,
    ) -> Tuple[List, List]:
        """Calculates radial intensities and yields the corresponding polar representation

        :param only_rad_plot: If only the radial intensity profile (black) is to be plotted or a version of the image with highlightes contour (color map) alongside the colored radial intensity profile
        :type only_rad_plot: Optional[bool]
        :param show: If the plot is to be shown
        :type show: Optional[bool]
        :param save: If the plot is to be saved
        :type save: Optional[bool]
        :returns: List of angles with list of corresponding radial intensities
        :rtype: List, List
        """

        contours, _ = self.get_contours(self.binary_otsu, show=False)
        X, Y = self.get_coords_from_contours(contours)
        cx, cy = self.get_center_coords()

        lines = [
            ([int(cx), x], [int(cy), y]) for (x, y) in zip(X, Y)
        ]  # all lines connecting the center coordinate to each coordinate of the identified contour
        line_lens = []

        intensities = []
        colors = cm.jet(np.linspace(0, 1, len(list(lines))))  # color map

        for (color, line) in zip(colors, lines):
            xx, yy = line
            line_lens.append(
                (np.sqrt(np.abs(xx[1] - xx[0]) ** 2 + np.abs(yy[1] - yy[0]) ** 2))
            )  # length of each line
            intensities.append(
                np.sum(profile_line(self.gray, (xx[0], yy[0]), (xx[1], yy[1])))
            )  # sum of pixel values along line

        if int(0.02 * len(intensities)) % 2 == 0:
            window_length = int(0.02 * len(intensities)) + 1
        else:
            window_length = int(0.02 * len(intensities))

        _intensities = [
            (a / b) for (a, b) in zip(intensities, line_lens)
        ]  # normalization with line length
        intensities = _intensities

        # smoothing of data
        try:
            intensities = savgol_filter(intensities, window_length, 3)
        except ValueError:
            print("\n\t\tValueError encountered in savgol filter")
            return None
        intensities /= np.max(intensities)  # normalization to a maximum of 1

        deg = 2 * np.pi / (len(intensities))
        degs = [
            x * deg for x, xx in enumerate(intensities)
        ]  # calculate angle of each pixel of contour
        degs_p = [
            np.angle((xx - cx) + 1j * (yy - cy)) + np.pi for (xx, yy) in zip(X, Y)
        ]  # calculate angle of each pixel of contour

        def shift_cmap(color_arr: np.ndarray) -> List:
            """For unknwon reasons there is a shift introduced somewhere. This function maps the first coordinate/pixel/anlge to the first color. Only works for BFP images and might need to be modified"""
            deg_00 = np.rad2deg(np.arctan2((cx - X[0]), (cy - Y[0])))
            add_shift = 2 * deg_00 / 360

            color_arr = list(color_arr)
            col0 = color_arr[: int(3 * len(color_arr) / 4)]
            col1 = color_arr[int(3 * len(color_arr) / 4) :]
            color_arr = col1 + col0

            col0 = color_arr[: int(add_shift * len(color_arr))]
            col1 = color_arr[int(add_shift * len(color_arr)) :]

            return col1 + col0

        if only_rad_plot == True:
            fig = plt.figure(9009, figsize=(6, 6))
            ax0 = fig.add_subplot(111, projection="polar")
            for i, (deg, intensity) in enumerate(zip(degs_p, intensities)):
                ax0.plot(deg, intensity, marker=".", color="black")
            ax0.set_theta_zero_location("N")

            if save == True:
                plt.savefig(f"{self.name}_radial_profile.{self.out_format}", dpi=dpi)

        elif only_rad_plot == False:
            fig = plt.figure(8008, figsize=(12, 6))
            ax0 = fig.add_subplot(121)

            ax0.plot(cx, cy, marker="X", markersize=10, color="yellow", label="Center")
            for (xx, yy, color) in zip(X, Y, colors):
                ax0.plot(xx, yy, marker="o", color=color, markersize=5)
                # break
            # ax0.gca().invert_yaxis()
            ax0.axis("off")
            ax0.imshow(
                self.gray, cmap="gray", vmin=np.amin(self.gray), vmax=np.amax(self.gray)
            )

            colors_r = cm.jet_r(np.linspace(0, 1, len(list(lines))))
            colors_r = shift_cmap(colors_r)
            ax1 = fig.add_subplot(122, projection="polar")
            for i, (deg, intensity, color) in enumerate(
                zip(degs_p, intensities, colors_r)
            ):
                ax1.plot(deg, intensity, marker=".", color=color)
                # break
            ax1.set_theta_zero_location("N")

            plt.tight_layout()
            plt.subplots_adjust(left=+0.05, bottom=0.1, right=0.85, top=0.9, wspace=0.2)

            if save == True:
                plt.savefig(f"{self.name}_radial_profile_2.{self.out_format}", dpi=dpi)

        if show == True:
            plt.show()

        else:
            plt.close()

        return degs, intensities

    def get_radial_slice_intensities(
        self,
        threshold: float,
        only_rad_plot: Optional[bool] = True,
        show: Optional[bool] = True,
        save: Optional[bool] = True,
    ):
        """Calculates radial slice intensities and yields the corresponding polar representation. Based on radial intensity algorithm.


        :param threshold: Threshold in percent of total radial line length, at which the slices should start
        :type threshold: float
        :param only_rad_plot: If only the radial intensity profile (black) is to be plotted or a version of the image with highlightes contour (color map) alongside the colored radial intensity profile
        :type only_rad_plot: Optional[bool]
        :param show: If the plot is to be shown
        :type show: Optional[bool]
        :param save: If the plot is to be saved
        :type save: Optional[bool]
        :returns: List of angles with list of corresponding radial intensities
        :rtype: List, List
        """

        contours, _ = self.get_contours(self.binary_otsu, show=False)
        X, Y = self.get_coords_from_contours(contours)
        cx, cy = self.get_center_coords()

        lines = [([int(cx), x], [int(cy), y]) for (x, y) in zip(X, Y)]
        line_lens = []

        intensities = []
        intersections = []
        colors = cm.jet(np.linspace(0, 1, len(list(lines))))

        for line in lines:
            xx, yy = line

            prof_line_whole = np.array(
                profile_line(self.gray, (xx[0], yy[0]), (xx[1], yy[1]))
            )
            prof_line = prof_line_whole[int(threshold * len(prof_line_whole)) :]

            shapely_circle = Point(xx[1], yy[1]).buffer(len(prof_line))
            shapely_line = LineString([[xx[0], yy[0]], [xx[1], yy[1]]])
            intersect = shapely_line.intersection(shapely_circle)
            intersect_x, intersect_y = (
                intersect.coords.xy[0][0],
                intersect.coords.xy[1][0],
            )
            intersections.append([intersect_x, intersect_y])

            prof_line = np.array(
                profile_line(self.gray, (intersect_x, intersect_y), (xx[1], yy[1]))
            )
            intensities.append(np.sum(prof_line))
            line_lens.append(
                (
                    np.sqrt(
                        np.abs(xx[1] - intersect_x) ** 2
                        + np.abs(yy[1] - intersect_y) ** 2
                    )
                )
            )

        if int(0.02 * len(intensities)) % 2 == 0:
            window_length = int(0.02 * len(intensities)) + 1
        else:
            window_length = int(0.02 * len(intensities))

        try:
            intensities = savgol_filter(intensities, window_length, 3)
        except ValueError:
            print("ValueError encountered in savgol filter")
            return None

        z2polar = lambda z: np.angle(z)

        deg = 2 * np.pi / (len(intensities))
        degs = [x * deg for x, xx in enumerate(intensities)]
        degs_p = [
            np.angle((xx - cx) + 1j * (yy - cy)) + np.pi for (xx, yy) in zip(X, Y)
        ]

        def shift_cmap(color_arr):
            deg_00 = np.rad2deg(np.arctan2((cx - X[0]), (cy - Y[0])))
            add_shift = 2 * deg_00 / 360

            color_arr = list(color_arr)
            col0 = color_arr[: int(3 * len(color_arr) / 4)]
            col1 = color_arr[int(3 * len(color_arr) / 4) :]
            color_arr = col1 + col0

            col0 = color_arr[: int(add_shift * len(color_arr))]
            col1 = color_arr[int(add_shift * len(color_arr)) :]

            return col1 + col0

        if only_rad_plot == True:
            fig = plt.figure(9009, figsize=(6, 6))
            ax0 = fig.add_subplot(111, projection="polar")
            for i, (deg, intensity) in enumerate(zip(degs_p, intensities)):
                ax0.plot(deg, intensity, marker=".", color="yellow")
            ax0.set_theta_zero_location("N")

            if save == True:
                plt.savefig(
                    f"{self.name}_radial_profile_thres_{threshold}.{self.out_format}",
                    dpi=dpi,
                )

        elif only_rad_plot == False:
            fig = plt.figure(8008, figsize=(12, 6))
            ax0 = fig.add_subplot(121)

            ax0.plot(cx, cy, marker="X", markersize=10, color="yellow", label="Center")
            for (xx, yy, pp, color) in zip(X, Y, intersections, colors):
                ax0.plot(xx, yy, marker="o", color=color, markersize=5)
                ax0.plot(pp[0], pp[1], marker="o", color=color, markersize=5)

            # ax0.gca().invert_yaxis()
            ax0.axis("off")
            ax0.imshow(
                self.gray, cmap="gray", vmin=np.amin(self.gray), vmax=np.amax(self.gray)
            )

            colors_r = cm.jet_r(np.linspace(0, 1, len(list(lines))))
            colors_r = shift_cmap(colors_r)
            ax1 = fig.add_subplot(122, projection="polar")
            for i, (deg, intensity, color) in enumerate(
                zip(degs_p, intensities, colors_r)
            ):
                ax1.plot(deg, intensity, marker=".", color=color)
                # break
            ax1.set_theta_zero_location("N")

            plt.tight_layout()
            plt.subplots_adjust(left=+0.05, bottom=0.1, right=0.85, top=0.9, wspace=0.2)

            if save == True:
                plt.savefig(
                    f"{self.name}_radial_profile_thres_{threshold}_2.{self.out_format}",
                    dpi=dpi,
                )

        if show == True:
            plt.show()

        else:
            plt.close()

        return degs, intensities
