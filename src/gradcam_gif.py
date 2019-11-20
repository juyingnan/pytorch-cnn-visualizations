"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import torch
import sys
import os
import numpy as np
import imageio

from misc_functions import (get_example_params,
                            get_example_by_filename,
                            save_gif_images,
                            convert_to_grayscale,
                            )
from gradcam import GradCam
from guided_backprop import GuidedBackprop
from guided_gradcam import guided_grad_cam
import sys
import os
from torch.autograd import Variable
from torchvision import models, transforms


def predict_image(image, pretrained_model):
    # print("Prediction in progress")

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = pretrained_model(input)

    index = output.data.numpy().argmax()

    return index


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
        if file_class == int(file_name[0]):

            gif_img_list = list()

            for epoch in range(15):
                pretrained_model = torch.load(
                    r'C:\Users\bunny\PycharmProjects\KaggleTest\classification\model_ship_{}.pth'.format(epoch + 1),
                    map_location=lambda storage, loc: storage)

                # Grad cam
                gcv2 = GradCam(pretrained_model, target_layer=11)
                # Generate cam mask
                cam = gcv2.generate_cam(prep_img, target_class)
                # print('Grad cam completed')

                # Guided backprop
                GBP = GuidedBackprop(pretrained_model)
                # Get gradients
                guided_grads = GBP.generate_gradients(prep_img, target_class)
                # print('Guided backpropagation completed')

                # Guided Grad cam
                cam_gb = guided_grad_cam(cam, guided_grads)
                # save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
                # print('Guided grad cam completed')

                # prediction
                classes = ('no', 'yes')
                pred_index = predict_image(original_image, pretrained_model)
                # print(classes[pred_index])

                # Save mask
                merged = save_gif_images(original_image, cam, cam_gb, classes[pred_index],
                                         file_name_to_export + f'_{epoch + 1}', color_map='jet') # cividis
                gif_img_list.append(merged)
                print(f'Grad cam gif completed for {file_name} : {file_class} in epoch {epoch + 1}')

            for rep in range(5):
                gif_img_list.append(gif_img_list[-1])
            # imageio.mimsave('../results/test.gif', gif_img_list, duration=0.2)
            gif_img_list[0].save(f'../results/{file_name_to_export}.gif', format='GIF', append_images=gif_img_list[1:],
                                 save_all=True, duration=200, loop=0)
