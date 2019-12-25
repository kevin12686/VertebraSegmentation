from os import mkdir, listdir
from os.path import splitext, exists, join
from PIL import Image, ImageOps
from tqdm import tqdm

SOURCE_DIR = [join("original_data", "f01"), join("original_data", "f02")]
TARGET_DIR = "extend_dataset"
SUB_DIR = ["image", "label"]

ROTATION_ANGLE = [180]

if __name__ == '__main__':
    if not exists(TARGET_DIR):
        mkdir(TARGET_DIR)
    for sub_dir in SUB_DIR:
        dir = join(TARGET_DIR, sub_dir)
        if not exists(dir):
            mkdir(dir)

    for source in SOURCE_DIR:
        for sub_dir in SUB_DIR:
            for file in tqdm(listdir(join(source, sub_dir)), desc=f"{source}\\{sub_dir}"):
                img = Image.open(join(source, sub_dir, file))
                img.save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_0.png"))
                for angle in ROTATION_ANGLE:
                    img.rotate(angle, expand=True).save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_{angle}.png"))
                img = ImageOps.mirror(img)
                img.save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f0.png"))
                for angle in ROTATION_ANGLE:
                    img.rotate(angle, expand=True).save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f{angle}.png"))
