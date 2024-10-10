import os
import numpy as np
from concave_hull import concave_hull_indexes
from bresenham import bresenham
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tqdm import tqdm
import tifffile

# ---
# Setup, define some variables and settings
# ---
image_width, image_height = 300, 300 # the number of pixels in X and Y, respectively
pixel_size = 0.102 # in microns

# ---
# Functions
# ---
def get_bresenham_points(data):
    data_connected = data.copy()
    data_connected = np.concatenate((data_connected, np.expand_dims(data_connected[0], axis=0)), axis=0) # add first point to close region
    roi = data.copy()
    for i in range(data_connected.shape[0] - 1):
        point1 = data_connected[i]
        point2 = data_connected[i + 1]
        bres_line2d = bresenham(point1[1], point1[0], point2[1], point2[0])
        intermediate_pixels = list(bres_line2d)
        intermediate_pixels_array = np.array(intermediate_pixels[1:-1]) # excluding first and last coords which are returned in the module
        intermediate_pixels_array = np.flip(intermediate_pixels_array, axis=1)
        roi = np.concatenate((roi, intermediate_pixels_array), axis=0)

    return roi


# ---
# Main
# ---
def main ():    
    Tk().withdraw()
    table_directory_path = askdirectory()
    print("Folder path containing tables: ", table_directory_path)

    temp_folder_path = os.path.join(table_directory_path, 'temp_images/')
    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)

    table_list = os.listdir(table_directory_path)
    table_list = [x for x in table_list if x.find('.csv') > 0]
    table_list.sort()
    print('Number of tables in path: ', str(len(table_list)))

    data_tosave = [[None, None, None]] * len(table_list)
    counter = 0 # should match the timepoint

    for table_name in tqdm(table_list):
        # print('Table name: ', table_name)
        
        table_data = []
        with open(os.path.join(table_directory_path, table_name)) as f:
            table_temp = f.read()
            table_temp = table_temp.split('\n')
            table_temp = table_temp[1:-1]
            table_temp = [x.split(',') for x in table_temp]
            table_temp = [x[1:] for x in table_temp]
            table_temp = [[int(x[0]), int(x[1])] for x in table_temp]
            table_data.append(table_temp)
        
        table_data = np.asarray(table_data).squeeze()
        
        blank_image = np.zeros((image_height, image_width))
        hull_shape_indexes = concave_hull_indexes(
            points=table_data,
            concavity=2.0, 
            length_threshold=0.0
            )
        
        table_data_vertices = table_data[hull_shape_indexes]
        shape_roi_coordinates = get_bresenham_points(table_data_vertices) 
        
        for coord in shape_roi_coordinates:
            blank_image[coord[1], coord[0]] = 1
        
        blank_image = binary_fill_holes(blank_image)
        blank_image = blank_image.astype(np.uint8)
        image_label = label(blank_image)
        stats = regionprops(image_label)

        area = stats[0].area
        area_scaled = area * pixel_size**2

        tifffile.imwrite(os.path.join(temp_folder_path, table_name[:table_name.index('.csv')]), blank_image, dtype='uint8')

        data_tosave[counter] = [counter, table_name, area_scaled]

        counter += 1
    
    with open(os.path.join(table_directory_path, 'concave_hull_area.csv'), 'w') as outfile:
        for outline in data_tosave:
            outfile.write(str(outline[0])+','+str(outline[1])+','+str(outline[2])+'\n')

    outfile.close()

main()