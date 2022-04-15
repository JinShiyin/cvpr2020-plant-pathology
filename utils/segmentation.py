from multiprocessing import Pool, Manager
from tkinter.messagebox import NO
import traceback
import cv2
import numpy as np
from matplotlib import pyplot
import os
import json

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

def segmentation(input_dir, in_filename, output_dir, debug=True):
    """generate the egg masks of samples. The masks are saved by 
    categories. 'egg_roi' saves the main region of egg. 

    Args:
        input_dir (_type_): _description_
        in_filename (_type_): _description_
        output_dir (_type_): _description_
        debug (bool, optional): _description_. Defaults to True.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
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
    filename = f'{filename}.png'
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
        
        

if __name__ == '__main__':
    # image_path =  '/Users/shandalau/Documents/Project/EggsCanding/2022-02-13-Eggs-Test/000513851_Egg1_(ok)_R_0_cam2.bmp'
    # image_path =  '/Users/shandalau/Documents/Project/EggsCanding/2022-02-13-Eggs-Test/153729136_Egg3_(ruopei--ok)_R_0_cam4.bmp'
    base_output_dir = 'datasets/2022-03-17-Egg-Masks'
    os.makedirs(base_output_dir, exist_ok=True)
    # filenames = ['153729136_Egg3_(ruopei--ok)_R_0_cam4.bmp', '174525346_Egg6_(sipei)_L_0_cam5.bmp']
    # filenames = ['174525346_Egg6_(sipei)_L_0_cam5.bmp']
    with Pool(8) as pool:
        input_dir = '/Users/shandalau/Documents/Project/EggsCanding/datasets/2022-03-02-Eggs'
        manager = Manager()
        roi_dict = {} # 读取图像
        results = []
        classnames = [classname for classname in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, classname))]
        print(classnames)
        for classname in classnames:
            class_dir = os.path.join(input_dir, classname)
            filenames = os.listdir(class_dir)
            output_dir = os.path.join(base_output_dir, classname) 
            os.makedirs(output_dir, exist_ok=True)
            for filename in filenames:
                result = pool.apply_async(segmentation, args=(class_dir, filename, output_dir,))
                results.append(result)
            for result in results:
                roi_dict.update(result.get())
            with open(os.path.join(output_dir, 'roi.json'), 'w') as f:
                json.dump(roi_dict, f, indent=2)
        pool.close()
        pool.join()
    # img = cv2.imread(image_path)
    # img = cv2.pyrMeanShiftFiltering(img, 20, 40)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)