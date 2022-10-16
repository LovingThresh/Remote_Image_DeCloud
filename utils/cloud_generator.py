# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 13:35
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : cloud_generator.py
# @Software: PyCharm
# @From    : https://github.com/cidcom/SatelliteCloudGenerator
import os
import shutil

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.src import *
import imageio
import random

plt.rcParams["figure.figsize"] = (20, 10)
# Medium-wave infrared
mwir_img = imageio.imread('./utils/image_example/mwir_example.png')[:256, :256, 0] / 255
rgb_img = imageio.imread('./utils/image_example/rgb_example.png')[..., :3] / 255

clear_image, cloud_mask, shadows_mask = add_cloud_and_shadow(cv2.resize(rgb_img, (2048, 2048)),
                                                             return_cloud=True
                                                             )
# Cloud and Shadows
plt.subplot(1, 4, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 4, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 4, 3)
plt.imshow(cloud_mask)
plt.title('Channel-wise Cloud Mask')
plt.subplot(1, 4, 4)
plt.imshow(shadows_mask)
plt.title('Channel-wise Shadow Mask')
plt.show()

# Shadows
clear_image, cloud_mask, shadows_mask = add_cloud_and_shadow(rgb_img,
                                                             max_lvl=1.0,
                                                             min_lvl=1.0,
                                                             decay_factor=0.5,
                                                             return_cloud=True
                                                             )

plt.subplot(1, 4, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 4, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 4, 3)
plt.imshow(cloud_mask)
plt.title('Channel-wise Cloud Mask')
plt.subplot(1, 4, 4)
plt.imshow(shadows_mask)
plt.title('Channel-wise Shadow Mask')
plt.show()

# Thick Cloud
clear_image, cloud_mask = add_cloud(cv2.resize(rgb_img, (4096, 1024)),
                                    min_lvl=0.2,
                                    max_lvl=1.0,
                                    clear_threshold=0.0,
                                    decay_factor=1.5,
                                    blur_scaling=0,
                                    const_scale=True,
                                    cloud_color=False,
                                    return_cloud=True
                                    )

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(cloud_mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

# Thick Foggy Cloud

clear_image, cloud_mask = add_cloud(rgb_img,
                                    min_lvl=0.1,
                                    max_lvl=0.5,
                                    decay_factor=1.85,
                                    return_cloud=True)

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(cloud_mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

# Thin Fog
clear_image, Fog_mask = add_cloud(rgb_img,
                                  min_lvl=0.2,
                                  max_lvl=1.0,
                                  decay_factor=1,
                                  return_cloud=True)

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(Fog_mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

# cloud color
clear_image, cloud_mask = add_cloud(rgb_img,
                                    min_lvl=0.0,
                                    max_lvl=1.0,
                                    cloud_color=False,
                                    channel_offset=0,
                                    blur_scaling=0,
                                    return_cloud=True
                                    )
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(clear_image)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(cloud_mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

cl, mask = add_cloud(rgb_img,
                     min_lvl=0.0,
                     max_lvl=1.0,
                     cloud_color=False,
                     channel_offset=3,
                     blur_scaling=0,
                     return_cloud=True
                     )

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(cl)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

cl, mask = add_cloud(rgb_img,
                     min_lvl=0.3,
                     max_lvl=1.0,
                     cloud_color=False,
                     channel_offset=0,
                     blur_scaling=0.0,
                     return_cloud=True
                     )

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(cl)
plt.title('Simulated')
plt.subplot(1, 3, 3)
plt.imshow(mask)
plt.title('Channel-wise Cloud Mask')
plt.show()

for i in range(1000):
    v_x = 192
    v_y = 192
    seed = random.randint(1, 4)
    if seed == 1:
        clear_image, cloud_mask = add_cloud(cv2.resize(rgb_img, (4096 - 512, 1024)),
                                            min_lvl=0.2,
                                            max_lvl=1.0,
                                            clear_threshold=0.0,
                                            decay_factor=1.5,
                                            blur_scaling=1.0,
                                            const_scale=True,
                                            cloud_color=False,
                                            return_cloud=True
                                            )
        cloud_mask_left = torch.zeros((1024, 256, 3))
        cloud_mask_right = torch.zeros((1024, 256, 3))
        cloud_mask = torch.concat([cloud_mask_left, cloud_mask, cloud_mask_right], dim=1)
        os.mkdir('crop_image_1024/{}_{}'.format(i, seed))
        for t in range(17):
            s = v_y * t
            crop_image = cloud_mask[0:1024, 0 + s: 1024 + s, :]
            crop_image = cv2.cvtColor(np.uint8(crop_image.numpy() * 255), cv2.COLOR_RGB2BGR)
            cv2.imwrite('crop_image_1024/{}_{}/{}s.png'.format(i, seed, t), crop_image)
    if seed == 2:
        clear_image, cloud_mask = add_cloud(cv2.resize(rgb_img, (1024, 4096 - 512)),
                                            min_lvl=0.15,
                                            max_lvl=1.0,
                                            clear_threshold=0.0,
                                            decay_factor=1.5,
                                            blur_scaling=1.0,
                                            const_scale=True,
                                            cloud_color=False,
                                            return_cloud=True
                                            )
        cloud_mask_left = torch.zeros((256, 1024, 3))
        cloud_mask_right = torch.zeros((256, 1024, 3))
        cloud_mask = torch.concat([cloud_mask_left, cloud_mask, cloud_mask_right], dim=0)
        os.mkdir('crop_image_1024/{}_{}'.format(i, seed))
        for t in range(17):
            s = v_x * t
            crop_image = cloud_mask[0 + s: 1024 + s, 0:1024, :]
            crop_image = cv2.cvtColor(np.uint8(crop_image.numpy() * 255), cv2.COLOR_RGB2BGR)
            cv2.imwrite('crop_image_1024/{}_{}/{}s.png'.format(i, seed, t), crop_image)
    if seed == 3:
        clear_image, cloud_mask = add_cloud(cv2.resize(rgb_img, (4096 - 512, 1024)),
                                            min_lvl=0.15,
                                            max_lvl=1.0,
                                            clear_threshold=0.0,
                                            decay_factor=1.5,
                                            blur_scaling=1.0,
                                            const_scale=True,
                                            cloud_color=False,
                                            return_cloud=True
                                            )
        cloud_mask_left = torch.zeros((1024, 256, 3))
        cloud_mask_right = torch.zeros((1024, 256, 3))
        cloud_mask = torch.concat([cloud_mask_left, cloud_mask, cloud_mask_right], dim=1)
        os.mkdir('crop_image_1024/{}_{}'.format(i, seed))
        for t in range(17):
            s = v_y * t
            crop_image = cloud_mask[0 : 1024, 3072 - s : 4096 - s, :]
            crop_image = cv2.cvtColor(np.uint8(crop_image.numpy() * 255), cv2.COLOR_RGB2BGR)
            cv2.imwrite('crop_image_1024/{}_{}/{}s.png'.format(i, seed, t), crop_image)
    if seed == 4:
        clear_image, cloud_mask = add_cloud(cv2.resize(rgb_img, (1024, 4096 - 512)),
                                            min_lvl=0.15,
                                            max_lvl=1.0,
                                            clear_threshold=0.0,
                                            decay_factor=1.5,
                                            blur_scaling=1.0,
                                            const_scale=True,
                                            cloud_color=False,
                                            return_cloud=True
                                            )
        cloud_mask_left = torch.zeros((256, 1024, 3))
        cloud_mask_right = torch.zeros((256, 1024, 3))
        cloud_mask = torch.concat([cloud_mask_left, cloud_mask, cloud_mask_right], dim=0)
        os.mkdir('crop_image_1024/{}_{}'.format(i, seed))
        for t in range(17):
            s = v_x * t
            crop_image = cloud_mask[3072 - s : 4096 - s, 0:1024, :]
            crop_image = cv2.cvtColor(np.uint8(crop_image.numpy() * 255), cv2.COLOR_RGB2BGR)
            cv2.imwrite('crop_image_1024/{}_{}/{}s.png'.format(i, seed, t), crop_image)

# Mix Function
cloud = 1 - mask
input = torch.FloatTensor(rgb_img)
cloud_base = torch.ones_like(input)
output = input * cloud + cloud_base * (1 - cloud)

# Thin cloud
# generate thin cloud

# 随机挑选训练图像 400 images, 验证图像50 images, 测试图像 50 images
image_files = os.listdir(r'O:\RICE1\label')
random.shuffle(image_files)
random.shuffle(image_files)
train_images_list = image_files[:400]
val_images_list = image_files[400:450]
test_image_list = image_files[450:]

for image in test_image_list:
    shutil.copyfile(r'O:\RICE1\label/' + image, r'O:\RICE1\test\image/' + image)
    shutil.copyfile(r'O:\RICE1\cloud/' + image, r'O:\RICE1\test\cloud_image/' + image)

for i in range(9):
    for image in train_images_list:
        rgb_img = imageio.imread(r'O:\RICE1\train\image/' + image)[..., :3] / 255
        cloud_image, Fog_mask = add_cloud(rgb_img,
                                          min_lvl=(0.5, 0.65),
                                          max_lvl=(0.7, 0.75),
                                          decay_factor=1,
                                          return_cloud=True)
        cloud_image = np.uint8(cloud_image * 255)
        cloud_image = cv2.cvtColor(cloud_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(r'O:\RICE1\train\cloud_image/' + '{}_'.format(i) + image, cloud_image)
        rgb_img = np.uint8(rgb_img * 255)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(r'O:\RICE1\train\image/' + '{}_'.format(i) + image, rgb_img)
    print(i)
