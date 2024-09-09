import os
import numpy as np
from scipy.stats import kurtosis
# import tifffile
from utils import *
from focus_metrics import *
from bioio import BioImage
import bioio_czi
from datetime import datetime
from tqdm import tqdm

folder_path = '/mnt/Data/nuno_martins/Image_data/NPM_Ex003_Re07/'
file_extension = '.czi'

def main(path, extension):
    filelist = os.listdir(path)
    filelist = [x for x in filelist if x.find(extension) > 0]

    number_data = []
    rist_row = ['name', 'max', 'min', 'mean', 'var', 'norm_var', 'kurtosis', 'brenner', 'abs_laplacian', 'sqr_laplacian', 'tv', 'block_tv', 'tenengrad', 'vollath_f4', 'vollath_f5', 'sym_vollath_f4']
    number_data.append(rist_row)

    for file in tqdm(filelist):
        numbers = []
        # print("name: ", file)
        numbers.append(file)

        #load image data middle z plane
        data = BioImage(path+file, reader=bioio_czi.Reader)
        img = data.data[:, 1, (data.dims.Z // 2), ...].squeeze() # shape: TCZYX

        #start analysis
        numbers.append(np.max(img))
        numbers.append(np.min(img))
        numbers.append(np.mean(img))
        numbers.append(np.std(img)**2)
        norm_var = np.var(img)/np.mean(img)**2
        numbers.append(norm_var)
        numbers.append(kurtosis(img, axis=None))


        #image quality metrics
        brenner_score = brenner(img)
        numbers.append(brenner_score)

        abs_laplacian = absolute_laplacian(img)
        numbers.append(abs_laplacian)

        sqr_laplacian = squared_laplacian(img)
        numbers.append(sqr_laplacian)

        tv = total_variation(img)
        numbers.append(tv)

        block_tv = block_total_variation(img)
        numbers.append(block_tv)

        tenengrad_score = tenengrad(img)
        numbers.append(tenengrad_score)

        vollath_f4_score = vollath_f4(img) 
        numbers.append(vollath_f4_score)

        vollath_f5_score = vollath_f5(img)
        numbers.append(vollath_f5_score)

        sym_vollath_f4 = symmetric_vollath_f4(img)
        numbers.append(sym_vollath_f4)

        number_data.append(numbers)
    
    date = datetime.today().strftime('%Y%m%d_%H%M%S')
    table_name = 'background_metrics'+date+'.csv'

    with open(os.path.join(path, table_name), 'w') as outfile:
        for row in number_data:
            output = ''
            for number in row:
                output += str(number)+','
            output += '\n'
            outfile.write("%s" % output)

    outfile.close()

main(folder_path, file_extension)