import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from natsort import natsorted
from typing import Optional, List
import pandas as pd

###################################################################################################################################################################
# Can be used to create corrected BFP images. Given as inspiration, has to be adapted to user's needs prior to running
###################################################################################################################################################################

plt.rc('font', size = 25)
figsize = (11, 8)
figsize_hist = (14, 8)
fontsize = 25
fontsize_legend = 15
fontsize_title = 25
markersize = 18
markersize_filled = 5
linewidth = 3
capsize = 10
binwidth = 0.01
dpi = 100
markers_filled = ['o', 'v', 's', '*', 'X', 'D', '>', 'p', 'd', '^', '<', 'h', '8']
markers_filigrane = ['x', '1', '+', '4', '.', '3', '*']
colors = ['black', 'mediumblue', 'dodgerblue', 'slategray', 'teal', 'darkgreen', 'yellowgreen', 'gold', 'saddlebrown']
cmap = cm.turbo
###################################################################################################################################################################
###################################################################################################################################################################


def csv_to_img(csv_file: str, csv_reference: str, out_format: str, show_init: Optional[bool] = False) -> None:
    '''Substracts the reference from the recorded image to create the resulting BFP image.
    Input parameters are:
    :csv_file: str, name of recorded image,
    :csv_reference: str, name of recorded reference image,
    :out_format: str, desired type of output image,
    :show_init: Optional[bool], by default "False", if "True" displays the recorded image, reference and resulting image
    :return: None
    '''
    filename = csv_file.split('.')[0]

    try:
        with open(csv_file, 'r') as f:
            raw = np.loadtxt(f, delimiter = ' ', dtype = float)

        if show_init == True:
            fig = plt.figure(1, figsize = figsize)
            ax0 = fig.addsubplot(131)
            ax0.set_title("Image")
            ax0.imshow(raw)
            
    except FileNotFoundError:
        print(f'The specified file "{csv_file}" was not found in the current workign directory. Program is terminated.')
        exit()

    try:
        with open(csv_reference, 'r') as f:
            try:
                ref = np.loadtxt(f, delimiter = ' ', dtype = float)
            except ValueError:
                ref = np.loadtxt(f, delimiter = ',', dtype = float)

        if show_init == True:
            ax1 = fig.add_subplot(132)                
            ax1.set_title("Reference")
            ax1.imshow(ref)

        try:
            arr = raw - ref
        except ValueError:
            ref = np.concatenate((ref, [np.zeros(len(ref[0]))]), axis = 0)
            arr = raw - ref

        if show_init == True:
            ax2 = fig.add_subplot(133)
            ax2.set_title("Image - Reference")
            ax2.imshow(arr)

            plt.show()

    except FileNotFoundError:
        print(f'The specified reference "{csv_reference}" was not found in the current workign directory. Continuing without reference.')
        
        arr = raw

    img_uint8 = np.array(arr).astype('uint8')
    # img_float = np.array(arr).astype('float64')


    uint8_name = f'{filename}_original.{out_format}'

    print(f'\t\tCreating "{uint8_name}"')
    cv.imwrite(uint8_name, img_uint8)


def make_all_pngs(subdirs: Optional[bool] = True, out_format: Optional[str] = 'png', remove_old_imgs: Optional[str] = False) -> None:
    '''Automatically searches in the current working directory, if desired also in present subdirectories, for files to create BFP images.
    Input parameters are:
    :subdirs: Optional[bool], by default "True", controls if subdirectories are searched
    :out_format: Optional[str], by default str, desired type of output image,
    :remove_old)imgs: Optional[str], by default "False", if "True" deletes all (!) present files of the type specified in out_format
    :return: None
    '''
    def remove_present_imgs() -> None:
        imgs = [os.remove(x) for x in os.listdir() if x.endswith(out_format)]

    def get_matching_csv_files() -> List:
        files = natsorted([x for  x in os.listdir() if x.endswith('.csv') and 'spectrum' not in x and 'Meas' in x])
        refs = natsorted([x for  x in os.listdir() if x.endswith('.csv') and 'spectrum' not in x and 'Ref' in x])

        matches = []
        for ffile in files:
            meas_split = ffile.split('_')
            meas_idx = [idx for idx, x in enumerate(meas_split) if 'Meas' in x][0]
            meas_split.pop(meas_idx)

            for ref in refs:
                ref_split = ref.split('_')
                ref_split.pop(meas_idx)
                if meas_split == ref_split:
                    _meas_num = ffile.split('_')[meas_idx]
                    _ref_num = ref.split('_')[meas_idx]

                    meas_num = float(('').join([x for x in _meas_num if x in '0123456789']))
                    ref_num = float(('').join([x for x in _ref_num if x in '0123456789']))

                    if meas_num == ref_num:
                        matches.append([ffile, ref])

                    # else:
                    #     print(f'\n\t\tNo reference was found for {ffile}\n')

        return matches

    root = os.getcwd()
    dirs = [root]

    if subdirs == True:
        sub_dirs = [os.path.join(root, x) for x in os.listdir() if os.path.isdir(x) and 'Field' in x]
        dirs += sub_dirs

    for dir in dirs:
        print(f'\n\tIn {dir}:')
        os.chdir(dir)
        if remove_old_imgs == True:
            remove_present_imgs()

        matches = get_matching_csv_files()

        for match in matches:
            meas, ref = match
            csv_to_img(csv_file = meas, csv_reference = ref, out_format = out_format, show_init = False)

        os.chdir(root)


def make_all_pngs_no_ref(subdirs: Optional[bool] = True, out_format: Optional[str] = 'png', remove_old_imgs: Optional[str] = False) -> None:
    '''Automatically searches in the current working directory, if desired also in present subdirectories, for files to create BFP images.
    Input parameters are:
    :subdirs: Optional[bool], by default "True", controls if subdirectories are searched
    :out_format: Optional[str], by default str, desired type of output image,
    :remove_old)imgs: Optional[str], by default "False", if "True" deletes all (!) present files of the type specified in out_format
    :return: None
    '''
    def remove_present_imgs() -> None:
        imgs = [os.remove(x) for x in os.listdir() if x.endswith(out_format)]

    def get_csv_files() -> List:
        return  natsorted([x for  x in os.listdir() if x.endswith('.csv') and 'spectrum' not in x and 'meas' in x])
        
    def get_blank_ref(csv_file: str) -> str:
        df = pd.read_csv(csv_file, delimiter = ' ', header = None, index_col = False).to_numpy()
        blank = np.zeros(df.shape[:])

        df = pd.DataFrame(blank)
        df.to_csv("blank_ref.csv", header = False, index = False)
        return "blank_ref.csv"

    root = os.getcwd()
    dirs = [root]

    if subdirs == True:
        sub_dirs = [os.path.join(root, x) for x in os.listdir() if os.path.isdir(x) and 'Field' in x]
        dirs += sub_dirs

    for dir in dirs:
        print(f'\n\tIn {dir}:')
        os.chdir(dir)
        if remove_old_imgs == True:
            remove_present_imgs()

        ffiles = get_csv_files()
        for ffile in ffiles:
            ref = get_blank_ref(ffile)
            csv_to_img(csv_file = ffile, csv_reference = ref, out_format = out_format, show_init = False)

        os.chdir(root)



def runner():
    print('\nA process to create images from present .csv files was started.')

    io = input('Do you want to keep the default paramaters for this action (subdirs = True, out_format = "png", remove_old_imgs = False)? [y/n]: ')
    if io == 'n':
        print('\nPlease specify the following parameters:')
        subdirs = input('\tsubdirs [True/False]: ')
        if subdirs == "True":
            subdirs = True
        elif subdirs == "False":
            subdirs = False
        else:
            print('Wrong input, aborting.')
            exit()
        # print(subdirs, type(subdirs))

        out_format = str(input('\tout_format [png/tif/jpg]: '))
        if out_format not in ["png", "tif", "jpg"]:
            print('Wrong input, aborting.')
            exit()
        # print(out_format, type(out_format))

        remove_old_imgs = input('\tremove_old_imgs [True/False]: ')
        if remove_old_imgs == "True":
            remove_old_imgs = True
        elif remove_old_imgs == "False":
            remove_old_imgs = False
        else:
            print('Wrong input, aborting.')
            exit()
        # print(remove_old_imgs, type(remove_old_imgs))

    elif io == 'y': 
        subdirs, out_format, remove_old_imgs = True, 'png', False

    else:
        print('Wrong input, aborting.')
        exit()

    check = input('Are there references available? [y/n]: ')
    if check == 'y':
        print('\nCreating images...')
        make_all_pngs(subdirs, out_format, remove_old_imgs)

    elif check == 'n':
        print('\nCreating images without references...')
        make_all_pngs_no_ref(subdirs, out_format, remove_old_imgs)
        

if __name__ == '__main__':
    runner()
