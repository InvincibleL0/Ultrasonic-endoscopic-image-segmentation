import numpy as np
import cv2

def save_images(input, output1, output2, input_path, image_path, max_samples=4):
    # 在图片宽度上concatenate=>[batch_size,image_w,image_h * 2,image_c]
    image = np.concatenate([output1, output2], axis=2)  # concat 4D array, along width.
    # 纵向concatenate的图片个数=min(max_samples,batch_size)
    # image.shape[0]图片垂直尺寸，image.shape[1]图片水平尺寸，image.shape[2]图片通道数
    if max_samples > int(image.shape[0]):
        max_samples = int(image.shape[0])
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)

    # save image.
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

    # save input
    if input is not None:
        input_data = input[0:max_samples, :, :, :]
        input_data = np.concatenate([input_data[i, :, :, :] for i in range(max_samples)], axis=0)
        cv2.imwrite(input_path, np.uint8(input_data.clip(0., 1.) * 255.))
