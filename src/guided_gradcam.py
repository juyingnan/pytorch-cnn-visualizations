"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from misc_functions import (get_example_params,
                            get_example_by_filename,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop
import sys
import os


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    # Get params
    # target_example = 0  # Snake
    file_name_list = list()
    file_class_list = list()

    # file_name = '01_299.png'
    # file_class = 1
    if len(sys.argv) >= 3:
        file_name_list.append(sys.argv[1])
        file_class_list.append(int(sys.argv[2]))
    else:
        potential_file_list = os.listdir("../input_images/")
        for file_class in [0, 1]:
            for file_name in potential_file_list:
                if file_name.startswith(('00_', '01_', '10_', '11_')):
                    file_name_list.append(file_name)
                    file_class_list.append(file_class)

    for file_name, file_class in zip(file_name_list, file_class_list):
        (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
            get_example_by_filename(file_name, file_class)
        # get_example_params(target_example)

        # Grad cam
        gcv2 = GradCam(pretrained_model, target_layer=11)
        # Generate cam mask
        cam = gcv2.generate_cam(prep_img, target_class)
        print('Grad cam completed')

        # Guided backprop
        GBP = GuidedBackprop(pretrained_model)
        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_class)
        print('Guided backpropagation completed')

        # Guided Grad cam
        cam_gb = guided_grad_cam(cam, guided_grads)
        save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
        print('Guided grad cam completed')
