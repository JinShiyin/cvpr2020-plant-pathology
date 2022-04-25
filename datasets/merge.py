import os
import shutil

def merge(old_dir, new_dir):
    for filename in os.listdir(old_dir):
        filepath = os.path.join(old_dir, filename)
        if os.path.isdir(filepath):
            class_copy_dir = os.path.join(new_dir, filename)
            if not os.path.exists(class_copy_dir):
                os.makedirs(class_copy_dir, exist_ok=True)
            for class_filename in os.listdir(filepath):
                src = os.path.join(filepath, class_filename)
                target = os.path.join(class_copy_dir, class_filename)
                shutil.copyfile(src, target) 
                if not os.path.exists(target):
                    print(f'Copy {src} to {target}')
        else:
            shutil.copy2(filepath, new_dir)
            print(f'Copy {filepath} to {new_dir}')

if __name__ == '__main__':
    old_dir = '/Users/shandalau/Documents/Datasets/EggCanding/2022-04-15-Eggs'
    new_dir = '/Users/shandalau/Documents/Datasets/EggCanding/2022-04-18-Eggs'
    merge(old_dir, new_dir)
        