import os
from PIL import Image

if __name__ == '__main__':
    # 数据集所在目录
    data_root = "D:/Code/Intestinal ultrasound/datasets"

    # 将train目录下文件名保存进image_names列表中
    image_names = os.listdir(os.path.join(data_root, "train"))

    # 分别处理train目录和train_masks目录下的文件
    for filename in ["train", "train_masks"]:
        for image_name in image_names:
            # train目录下的图片
            if filename is "train":
                # 得到每张训练原图片的文件名路径 e.g. ../datasets/train/0.png ...
                image_file = os.path.join(data_root, filename, image_name)

                # PIL的Image类读取图像
                # convert()函数，用于不同模式图像之间的转换,L表示灰度转换为灰度图像,RGB表示彩色图像
                image = Image.open(image_file)

                # 创建../datasets/UnetImages/train/
                if not os.path.exists(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename)):
                    os.makedirs(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename))
                # 保存图片路径../datasets/UnetImages/train_masks/image_name
                image.save(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename, image_name))

            if filename is "train_masks":
                # 得到每张训练mask图片的文件名路径：e.g. ../datasets/train_masks/0_mask.png
                image_file = os.path.join(data_root, filename, image_name[:-4] + "_mask.png")
                image = Image.open(image_file)

                # 创建../datasets/UnetImages/train_mask/
                if not os.path.exists(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename)):
                    os.makedirs(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename))

                # 保存图片路径：../datasets/UnetImages/train_masks/image_name
                image.save(os.path.join("D:/Code/Intestinal ultrasound/datasets/UnetImages", filename,
                                        image_name[:-4] + "_mask.png"))

