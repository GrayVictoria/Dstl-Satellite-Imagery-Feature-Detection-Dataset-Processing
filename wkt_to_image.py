import csv
import os
from osgeo import gdal
import numpy as np
from shapely.geometry import MultiPolygon
import shapely.wkt as shapelywkt
import cv2
csv.field_size_limit(500*1024*1024)

img_dir_path = './three_band//'
grid_path = './grid_sizes.csv'
wkt_path = './train_wkt_v4.csv'


def to_ind(x): 
    return np.array(list(x)).astype(np.float32)

def generate_contours(polygon_list, img_size, xymax):
    '''
    Convert shapely MultipolygonWKT type of data (relative coordinate) into
    list type of date for polygon raster coordinates
    :param polygon_list:
    :param img_size:
    :param xymax:
    :return:
    '''
    if len(polygon_list) == 0:
        return [], []

    # def to_ind(x): return np.array(list(x)).astype(np.float32)

    perim_list = [convert_coordinate_to_raster(to_ind(poly.exterior.coords[:]), img_size, xymax) for poly in polygon_list]
    inter_list = [convert_coordinate_to_raster( to_ind(poly.coords[:]), img_size, xymax) for poly_ex in polygon_list for poly in poly_ex.interiors]
    # print(perim_list)
    return perim_list, inter_list


def convert_coordinate_to_raster(coords, img_size, xymax):
    '''
    Converts the relative coordinates of contours into raster coordinates.
    :param coords:
    :param img_size:
    :param xymax:
    :return:
    '''
    xmax, ymax = xymax
    width, height = img_size

    coords[:, 0] *= (height + 1) / xmax
    coords[:, 1] *= (width + 1) / ymax

    coords = np.round(coords).astype(np.int32)

    return coords


def generate_mask_from_contours(img_size, perim_list, inter_list, class_id=1):
    '''
    Create pixel-wise mask from contours from polygon of raster coordinates
    :param img_size:
    :param perim_list:
    :param inter_list:
    :param class_id:
    :return:
    '''
    mask = np.zeros(img_size, np.uint8)

    if perim_list is None:
        return mask
    # mask should match the dimension of image
    # however, cv2.fillpoly assumes the x and y axes are oppsite between mask and
    # perim_list (inter_list)
    cv2.fillPoly(mask, perim_list, class_id)
    cv2.fillPoly(mask, inter_list, 0)

    return mask

img_list = os.listdir(img_dir_path)
for img_name in img_list:
    if img_name[-4:]!='.tif':
        continue
    img = gdal.Open(img_dir_path+img_name)
    imgx = img.RasterXSize
    imgy = img.RasterYSize
    imgb = img.RasterCount
    
    img_name = img_name[:-4]
    label = np.zeros([imgx,imgy],np.uint8)

    with open(grid_path) as grid:
        grid_csv = csv.reader(grid)
        for row in grid_csv:
            image_name = row[0]
            if image_name==img_name:
                grid_x = row[1]
                grid_y = row[2]
                break
    flag = 0
    with open(wkt_path) as wkt:
        wkt_csv = csv.reader(wkt)
        for row in wkt_csv:
            # print(img_name)
            if row[0] == img_name:
                # print(row[0])
                flag = 1
                class_num = row[1]
                # print(class_num)
                print(imgx,imgy)
                plygon = row[2]
                polygon_list = shapelywkt.loads(plygon)
                perim_list, inter_list = generate_contours(
                    polygon_list, [float(imgx), float(imgy)], [float(grid_x), float(grid_y)])
                # print(len(perim_list))
                class_label = generate_mask_from_contours(
                    [int(imgx), int(imgy)], perim_list, inter_list, class_id=1)
                class_label = class_label*int(class_num)
                label = label+class_label
                # print(label.max())
                # print(np.unique(label))
                cv2.imwrite('d:/forest/data2/'+img_name+'_'+str(class_num)+'.bmp',class_label)
        # print(label.max())
    if flag ==1:
        exit()
#         class_num = wkt_csv[1]
#         print(image_name)

