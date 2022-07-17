from BFPy import Analysis #import the BFPy module. Pay attention to its location on your drive
import os
import numpy as np
import shutil


##############################################################################################################################
##############################################################################################################################
# Minimum working example for BFPy library
# Running this code might take quite some time, depending on the computatinal power of your machine
# All essential functionalities are implemented in this MWE
##############################################################################################################################
##############################################################################################################################

if __name__ == "__main__":
    if os.path.isdir('Example_data'): #check if directory with example data exists
        os.chdir('Example_data')
        root = os.getcwd()  

    else:
        print(f"The directory 'Example_data' does not exist. Aborting.")
        exit()

    dirs = [os.path.join(root, x) for x in os.listdir() if os.path.isdir(x) and 'Field' in x]  #all directories containing the data
    dirs.append(root)

    print('\n\nStarting BFP evaluation...\n')
    for dir in dirs: #iterating over all directories
        os.chdir(dir)
        print(f'\tIn {dir}:')

        pngs = [x for x in os.listdir() if x.endswith('.png') and x.endswith('original.png') and not x.startswith('csv')] #all corrected gray scale uint8 bfp images, need to be created priorly
        for p, png in enumerate(pngs):
            print(f'\n\t\t{p + 1} of {len(pngs)}: Analyzing {png}')

            try:
                file = Analysis(png)  #instantiate object for each file

                
                masked_background, name_masked_background = file.mask_background(file.img_non_uint8, lower_thres = int(0.05 * np.amax(file.img_non_uint8)), upper_thres = int(0.95 * np.amax(file.img_non_uint8)), show_hist = False, show = False) # mask background of orignal 
                masked_background_superres, name_masked_background_superres = file.get_super_res(masked_background, model = 'EDSR_x3.pb', blur = True, show = False) # create superresolution version of masked background

                file.make_false_color_img(masked_background, name_masked_background)
                file.make_false_color_img(masked_background_superres, name_masked_background_superres)


                # now, radial intensity analysis
                superres = [x for x in os.listdir() if x.endswith('superres.png')]  # on superres images
                originals = [x for x in os.listdir() if x.endswith('original.png')] # on normal images

                # block below if superres is source of data
                if len(superres) > 0:
                    if not os.path.isdir("Radial_Intensities"):
                        os.mkdir("Radial_Intensities")

                    for sr in superres:
                        shutil.copy(sr, os.path.join("Radial_Intensities", sr))

                # block below if normal is source of data
                # elif len(originals) > 0:
                #     if not os.path.isdir("Radial_Intensities"):
                #         os.mkdir("Radial_Intensities")

                #     for oo in originals:
                #         shutil.copy(oo, os.path.join("Radial_Intensities", oo))

                else:
                    print('Neither original images nor images with superesolution were found. Aborting.')
                    exit()

                
                # creates new directory and copies images to be analyzed
                os.chdir("Radial_Intensities")
                for img in os.listdir():
                    file = Analysis(img, out_format = 'pdf')

                    # normal radial intensity profiles
                    file.get_radial_intensities(only_rad_plot = True, show = False)
                    file.get_radial_intensities(only_rad_plot = False, show = False)
                    
                    # radial slice intensity profiles with given thresholds
                    thresholds = [0.4, 0.6]
                    for thrs in thresholds:
                        file.get_radial_slice_intensities(threshold = thrs, only_rad_plot = True, show = False)
                        file.get_radial_slice_intensities(threshold = thrs, only_rad_plot = False, show = False)

            except:
                print("\t\tAn error occured. Skipping this file and moving it into quarantine.")
                if not os.path.isdir('Quarantine'):
                    os.mkdir('Quarantine')

                shutil.copy(png, os.path.join('Quarantine', png))
                continue
            





        
