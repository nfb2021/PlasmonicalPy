import os
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from collections import Counter
from tqdm import trange
import warnings
import shutil

aspect_ratio = 1.10330516


plt.rc('font', size = 15)
figsize = (11, 8)
fontsize = 15
dpi = 200
alpha = 0.7

bar_plot_dict = {"G6": ("red", "darkred", r"$\gamma$6"), "D2": ("blue", "darkblue", r"$\delta$2"), "D4": ("magenta", "purple", r"$\delta$4")}


def get_corr_factors(csv_file, specs):
    check = False
    voltage, wd, magn = specs
    spec_used = f'{voltage}kV_WD{wd}_{magn}x'

    csv = pd.read_csv(csv_file).set_index('settings')
    idxs = []
    for idx, row in csv.iterrows():
        idxs.append(idx)
        if idx == spec_used:
            check = True
            x_factor, x_std, y_factor, y_std = row.to_numpy()
            return x_factor, x_std, y_factor, y_std

    if check == False:
        print(f'There are no corresponding correction factors based on your input {spec_used}. Try one of the following: {idxs}. Aborting.')
        exit()


def get_coords_from_contours(contour_list):
    X, Y = [], []
    contours = contour_list[0]

    for px in contours:
        X.append(px[0][0])
        Y.append(px[0][1])

    return X, Y

def get_distances(lst_a, lst_b, m=2):
    counter = Counter(lst_a)
    _coords_out, _distances_out = [], []
    for key, value in counter.items():
        if value == 2:
            indices = [idx for idx, element in enumerate(lst_a) if element == key]
            aa, bb = [key, key], [lst_b[np.min(indices)], lst_b[np.max(indices)]]
            _coords_out.append([aa, bb])
            _distances_out.append(np.abs(bb[0] - bb[1]))

    # remove outliers x < (median - 2*std) and x > (median + 2*std)

    _distances_out = np.array(_distances_out)
    distances_out = _distances_out[abs(_distances_out - np.mean(_distances_out)) < m * np.std(_distances_out)]
    coords_out = [_coords_out[idx] for idx, x in enumerate(_distances_out) if x in distances_out]

    return np.array(distances_out).astype('float64'), coords_out    
    

def df_from_dict(dict):
    # as dict might have values of diffrent length, first step is to pad them to the same length using NaN
    # this is required, as a dataframe can only be created from a dict with values of the same length
    _val_length = 0
    _key = None
    for item in dict.items(): # find longest value and corresponding key
        key, val = item
        if len(val) > _val_length:
            _val_length = len(val)
            _key = key

    for item in dict.items():
        key, val = item
        if key != _key:
            pad = np.array([np.nan for x in range(_val_length - len(val))])
            dict[key] = np.concatenate((dict[key], pad))

    return pd.DataFrame.from_dict(dict) # return a the padded dataframe


def get_attr(sample_name):
    try:
        return bar_plot_dict[sample_name]
    except KeyError:
        return "gold", "gold", "Unknown label"


   

def plot_single_results():
    __not_evaluatable_counter = 0
    df_stats = pd.DataFrame(index = ['median_length [nm]', 'std_length [nm]', 'median_width [nm]', 'std_width [nm]'])
    dict_all = {}
    specs = input('Specify the accel. voltage [kV], working distance and magnification (separated by commas, no spaces): ').split(',')
    x_factor, x_std, y_factor, y_std = get_corr_factors('NB_XL30_calibration.csv', specs)


    for i in trange(len(rods), desc = 'Evaluating SEM images...'):
        img_ov = rods[i]
        l_color, w_color, _ = get_attr(img_ov.split("_")[0])

        gray = cv.imread(img_ov)
        try:
            img = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        except:
            print(img_ov)
            continue
        img_copy = img.copy()
        img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
        img = img[0 : 400, :]
        blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
        _, binary_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchies = cv.findContours(binary_otsu, mode = cv.RETR_EXTERNAL, method = cv.CHAIN_APPROX_NONE)


        blank = np.zeros(img.shape, dtype = 'uint8')
        blank = cv.cvtColor(blank, cv.COLOR_GRAY2BGR)
    

        _contour_len, _contour_idx = 0, 0
        if len(contours) > 1:
            for c, contour in enumerate(contours):
                if len(contour) > _contour_len:
                    _contour_len = len(contour)
                    _contour_idx = c

                # cv.drawContours(img_copy, contours[c], -1, (0, 0, 255), 2)
                # cv.imshow(f'{img_ov}: {c}', img_copy)
            # cv.waitKey(0)
            contours = [contours[_contour_idx]]

            

        xx, yy = get_coords_from_contours(contours)
        xmin, xmax, ymin, ymax = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        lengths_px, lengths_coords = get_distances(xx, yy)
        widths_px, widths_coords = get_distances(yy, xx)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings('error')

            try:
                fig = plt.figure(i, figsize = (12, 8))
                ax0 = fig.add_subplot(221)
                # ax0.set_title(img_ov)
                ax0.imshow(gray, aspect = aspect_ratio)
                img_width, img_height, _ = gray.shape[:]
                ax0.set_xlim(int(xmin - 0.05*img_width), int(xmax + 0.05*img_width))
                ax0.set_ylim(int(ymin - 0.05*img_height), int(ymax + 0.05*img_height))
                ax0.set_xlabel('x [px]', fontsize = fontsize)
                ax0.set_ylabel('y [px]', fontsize = fontsize)

                
                
                lengths_nm = lengths_px * y_factor
                median_lengths = round(np.median(lengths_nm), 2)
                std_lengths = round(np.std(lengths_nm), 2)

                for l_coords in lengths_coords:
                    ax0.plot(l_coords[0], l_coords[1], linestyle = '-', linewidth = '1',  color = l_color, alpha = alpha)

        
                ax2 = fig.add_subplot(222)
                ax2.set_title(f'Median = {median_lengths}nm, std = {std_lengths}nm')

                try:
                    ax2.hist(lengths_nm, bins = int(np.sqrt(len(lengths_nm))), color = l_color, edgecolor = 'black')
                except ValueError:
                    ax2.hist(lengths_nm, color = l_color, edgecolor = 'black')

                ax2.set_ylabel('Counts', fontsize = fontsize)
                ax2.set_xlabel('Lengths [nm]', fontsize = fontsize)

                
                ax1 = fig.add_subplot(223)
                ax1.imshow(gray, aspect = aspect_ratio)

                
                widths_nm = widths_px * x_factor
                median_widths = round(np.median(widths_nm), 2)
                std_widths = round(np.std(widths_nm), 2)
                for w_coords in widths_coords:
                    ax1.plot(w_coords[1], w_coords[0], linestyle = '-', linewidth = '1', color = w_color, alpha = alpha)

                ax1.set_xlim(int(xmin - 0.05*img_width), int(xmax + 0.05*img_width))
                ax1.set_ylim(int(ymin - 0.05*img_height), int(ymax + 0.05*img_height))
                ax1.set_xlabel('x [px]', fontsize = fontsize)
                ax1.set_ylabel('y [px]', fontsize = fontsize)

                ax3 = fig.add_subplot(224)
                ax3.set_title(f'Median = {median_widths}nm, std = {std_widths}nm')

                try:
                    ax3.hist(widths_nm, bins = int(np.sqrt(len(widths_nm))), color = w_color, edgecolor = 'black')
                except ValueError:
                    ax3.hist(widths_nm, color = w_color, edgecolor = 'black')

                ax3.set_ylabel('Counts', fontsize = fontsize)
                ax3.set_xlabel('Widths [nm]', fontsize = fontsize)



                plt.tight_layout()
                name = img_ov.split('.TIF')[0]
                df_stats[name] = [median_lengths, std_lengths, median_widths, std_widths]
                try:
                    plt.savefig(f'{name}_hist.pdf')
                except DeprecationWarning:
                    pass
                plt.savefig(f'{name}_hist.png', dpi = dpi)
                # plt.show()
                plt.close()

                sample = img_ov.split('uCcm-2')[0]
                if f'{sample}uCcm-2_widths [nm]' not in dict_all.keys():
                    dict_all[f'{sample}uCcm-2_widths [nm]'] = widths_nm
                else:
                    dict_all[f'{sample}uCcm-2_widths [nm]'] = np.concatenate((dict_all[f'{sample}uCcm-2_widths [nm]'], np.array(widths_nm)), axis = 0)


                if f'{sample}uCcm-2_lengths [nm]' not in dict_all.keys():
                    dict_all[f'{sample}uCcm-2_lengths [nm]'] = lengths_nm
                else:
                    dict_all[f'{sample}uCcm-2_lengths [nm]'] = np.concatenate((dict_all[f'{sample}uCcm-2_lengths [nm]'], np.array(lengths_nm)), axis = 0)

                    
            except RuntimeWarning:
                if not os.path.isdir('Not evaluatable'):
                    os.mkdir('Not evaluatable')
                shutil.move(img_ov, os.path.join('Not evaluatable', img_ov))
                __not_evaluatable_counter += 1
                continue


    df_all = df_from_dict(dict_all)
    df_all.to_csv('rod_data.csv', index = False)
    df_stats.to_csv('results.csv')

    if __not_evaluatable_counter > 0:
        print(f'\n\t{__not_evaluatable_counter} images could not be evaluated. You will find them in the directory "Not Evaluatable".\n')
    
    return df_stats, df_all










if __name__ == '__main__':
    rods = natsorted([x for x in os.listdir() if x.endswith('.TIF')])


    _, df_all = plot_single_results()
