from multiprocessing import Pool, Manager
from tkinter.messagebox import NO
import traceback
import cv2
import numpy as np
from matplotlib import pyplot
import os
import json
def segmentation(image_path):
    """select roi mask and air room mask using segmentation algorithm 
    Args:
        image_path (_type_): _description_
    """
    # 读取图像
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    smooth_img = cv2.pyrMeanShiftFiltering(rgb_image, 20, 40)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image_result = rgb_image.copy()
    # gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY)
 
    th = 10
    ret, thresh_image = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY)
    # ret, thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg_image = cv2.dilate(opening_image, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening_image, cv2.DIST_L2, 5)
    ret, sure_fg_image = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg_image = np.uint8(sure_fg_image)
    masked_img = cv2.subtract(sure_bg_image, sure_fg_image)
 
    # Marker labelling
    ret, markers_image = cv2.connectedComponents(sure_fg_image)
    # Add one to all labels so that sure background is not 0, but 1
    markers_image = markers_image + 1
    # Now, mark the region of unknown with zero
    markers_image[masked_img == 255] = 0
 
    markers_image = cv2.watershed(rgb_image_result, markers_image)
    rgb_image_result[markers_image == -1] = [255, 0, 0]
 
    # 显示图像
    dpi = 100
    pyplot.figure('Image Display', figsize=[9, 9])
    titles = ['Original Image', 'Smooth Image', 'Gray Image', 'Thresh Image', 'ForeGround Image', 'BackGround Image',
              'Markers Image', 'Result Image']
    images = [rgb_image, smooth_img, gray_image, thresh_image, sure_fg_image, sure_bg_image, markers_image, rgb_image_result]
    cols = 4
    rows = len(images) // 4 + 1
    for i in range(len(images)):
        pyplot.subplot(rows, cols, i + 1)
        pyplot.imshow(images[i], 'gray')
        pyplot.title(titles[i])
        pyplot.xticks([])
        pyplot.yticks([])
    # pyplot.show()
    filename = os.path.splitext(os.path.basename(image_path))[0]
    pyplot.savefig(f'{th}_Result_{filename}.png')
 
    # 根据用户输入保存图像
    # if ord("q") == (cv2.waitKey(0) & 0xFF):
        # 销毁窗口
        # pyplot.close('all')
    return
 
def detect_esplise(image):
    params = cv2.SimpleBlobDetector_Params()
    # Set Area filtering parameters
    # params.filterByArea = True
    # params.minArea = 100
    
    # Set Circularity filtering parameters
    # params.filterByCircularity = True
    # params.minCircularity = 0.5
    
    # Set Convexity filtering parameters
    # params.filterByConvexity = True
    # params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(image)
    
    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blobs

def get_max_connected_components(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    max_component = np.zeros(output.shape, dtype=np.uint8)
    # print(nb_components)
    if len(stats) <= 1:
        return max_component, [0, 0], [output.shape[1], output.shape[0]]
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    x = stats[max_label, cv2.CC_STAT_LEFT]
    y = stats[max_label, cv2.CC_STAT_TOP]
    w = stats[max_label, cv2.CC_STAT_WIDTH]
    h = stats[max_label, cv2.CC_STAT_HEIGHT]
    # roi = cv2.rectangle(rgb_image.copy(), (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    max_component[output == max_label] = 255
    return max_component, [int(x), int(y)], [int(x + w), int(y + h)]

def get_roi(image, mask):
    max_component, p_tl, p_br = get_max_connected_components(mask)
    # mask_image = cv2.bitwise_and(image, image, mask=max_component)
    cropped_roi = image[p_tl[1]:p_br[1], p_tl[0]: p_br[0]]
    roi = np.zeros(image.shape, dtype=np.uint8)
    roi[p_tl[1]:p_br[1], p_tl[0]:p_br[0]] = cropped_roi
    return cropped_roi, max_component, image, p_tl, p_br, roi

def segmentation(input_dir, in_filename, output_dir, debug=False):
    image_path = os.path.join(input_dir, in_filename)
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f'Empty image {input_dir}/{in_filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filename = os.path.splitext(in_filename)[0]

    egg_mask_out = os.path.join(output_dir, 'egg_mask')
    air_room_mask_out = os.path.join(output_dir, 'ar_mask')
    egg_roi_out = os.path.join(output_dir, 'egg_roi')
    # if os.path.exists(egg_roi_out):
        # print(f'Existing file {egg_roi_out}, return')
        # return
    ar_roi_out = os.path.join(output_dir, 'ar_roi')
    debug_out = os.path.join(output_dir, 'debug')
    out_dirs = [egg_mask_out, air_room_mask_out, egg_roi_out, ar_roi_out, debug_out]
    filename = f'{filename}.jpg'
    for out in out_dirs:
        os.makedirs(out, exist_ok=True)
    
    egg_mask_outpath = os.path.join(egg_mask_out, filename)
    air_room_mask_outpath = os.path.join(air_room_mask_out, filename)
    egg_roi_outpath = os.path.join(egg_roi_out, filename)
    ar_roi_outpath = os.path.join(ar_roi_out, filename)

    smooth_img = cv2.pyrMeanShiftFiltering(image, 20, 100)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rgb_image_result = rgb_image.copy()
    # gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY)
 
    # 基于分水岭算法的图像分割(Image Segmentation with Watershed Algorithm)
    _, egg_mask = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    _, air_room_mask = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    air_room_mask = cv2.bitwise_not(air_room_mask)
    # ret, thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # fg_mask = cv2.dilate(opening_image, kernel, iterations=3)
    cropped_egg_roi, egg_component, mask_image, egg_tl, egg_br, egg_roi = get_roi(image, egg_mask)
    cropped_air_room_roi, air_room_component, _, ar_tl, ar_br, air_room_roi = get_roi(image, air_room_mask)

    # noise removal
    # kernel = np.ones((3, 3), np.uint8)
    # opening_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area

    # circle = detect_esplise(gray_image)
    # 显示图像
    # dpi = 100
    # pyplot.figure('Image Display', figsize=[9, 9])
    # titles = ['Original Image', 'Smooth Image', 'Gray Image', 'Egg Mask', 'Airroom Mask', 'Egg Component', 'Airroom Component', 'Result Image', 'Egg ROI', 'Airroom ROI']
    if debug:
        whole = [image, smooth_img, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB), 
                    cv2.cvtColor(egg_mask, cv2.COLOR_GRAY2RGB) , 
                    cv2.cvtColor(air_room_mask,  cv2.COLOR_GRAY2RGB), 
                    cv2.cvtColor(egg_component, cv2.COLOR_GRAY2RGB), 
                    cv2.cvtColor(air_room_component, cv2.COLOR_GRAY2RGB), 
                    mask_image, egg_roi, air_room_roi]
        whole = cv2.hconcat(whole)
        whole = cv2.cvtColor(whole, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(debug_out, f'{filename[:-4]}.jpg'), whole)

    
    # cols = 4
    # rows = len(images) // 4 + 1
    # for i in range(len(images)):
    #     pyplot.subplot(rows, cols, i + 1)
    #     pyplot.imshow(images[i], 'gray')
    #     pyplot.title(titles[i])
    #     pyplot.xticks([])
    #     pyplot.yticks([])
    # # pyplot.savefig(f'{th}_Result_{filename}.png')
    # pyplot.savefig(os.path.join(debug_out, f'{filename[:-4]}.jpg'))
    # pyplot.close()

    egg_mask_outpath = os.path.join(egg_mask_out, filename)
    air_room_mask_outpath = os.path.join(air_room_mask_out, filename)
    egg_roi_outpath = os.path.join(egg_roi_out, filename)
    ar_roi_outpath = os.path.join(ar_roi_out, filename)

    cv2.imwrite(egg_mask_outpath, egg_component)
    cv2.imwrite(air_room_mask_outpath, air_room_component)
    cv2.imwrite(egg_roi_outpath, cv2.cvtColor(cropped_egg_roi, cv2.COLOR_RGB2BGR))
    cv2.imwrite(ar_roi_outpath, cv2.cvtColor(cropped_air_room_roi, cv2.COLOR_RGB2BGR))
    # pyplot.show()
    # 根据用户输入保存图像
    # if ord("q") == (cv2.waitKey(0) & 0xFF):
        # 销毁窗口
        # pyplot.close('all')
    print(f'Processed {image_path}')
    return {filename : {'egg_roi': [egg_tl[0], egg_tl[1], egg_br[0], egg_br[1]], 'air_room_roi': [ar_tl[0], ar_tl[1], ar_br[0], ar_br[1]]}}
        

def init_seg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with Pool(8) as pool:
        roi_dict = {} # 读取图像
        results = []
        classnames = [classname for classname in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, classname))]
        print(classnames)
        for classname in classnames:
            in_class_dir = os.path.join(input_dir, classname)
            filenames = [filename for filename in os.listdir(in_class_dir) if filename.endswith('.jpg')]
            out_class_dir = os.path.join(output_dir, classname) 
            os.makedirs(out_class_dir, exist_ok=True)
            for filename in filenames:
                result = pool.apply_async(segmentation, args=(in_class_dir, filename, out_class_dir,))
                results.append(result)
        for result in results:
            roi_dict.update(result.get())
        with open(os.path.join(output_dir, 'roi.json'), 'w') as f:
            json.dump(roi_dict, f, indent=2)
        pool.close()
        pool.join()

        

if __name__ == '__main__':
    output_dir = '/Users/shandalau/Datasets/EggCanding/2022-04-15-Egg-Masks'
    input_dir = '/Users/shandalau/Datasets/EggCanding/2022-04-15-Eggs'
    init_seg(input_dir, output_dir)
    # img = cv2.imread(image_path)
    # img = cv2.pyrMeanShiftFiltering(img, 20, 40)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)