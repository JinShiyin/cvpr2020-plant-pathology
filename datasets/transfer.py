import os
import shutil
import cv2

def transfer_bmp2jpg(input_dir):
    dirnames = os.listdir(input_dir)
    dirnames = [os.path.join(input_dir, name) for name in dirnames if os.path.isdir(os.path.join(input_dir, name))]
    for dirname in dirnames:
        filenames = os.listdir(dirname)
        for filename in filenames:
            if not filename.endswith('.bmp'):
                print(f'Filtering {filename}')
                continue
            preffix = os.path.splitext(filename)[0]
            file_path = os.path.join(dirname, filename)
            img = cv2.imread(os.path.join(dirname, filename))
            cv2.imwrite(os.path.join(dirname, f'{preffix}.jpg'), img)
            os.remove(file_path)
            print(f'transfferred {file_path} to jpg format.')

        