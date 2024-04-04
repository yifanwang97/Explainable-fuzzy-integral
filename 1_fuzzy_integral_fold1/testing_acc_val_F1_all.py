import sympy as sp
import torchvision.models as models
import torch
from torch import nn
# import torchvision.datasets as datasets
import os
# import torchvision.transforms as transforms
import torch.nn.functional as F
from data_loader import get_Dataloader
import json
from sklearn.metrics import f1_score

def get_info_of_dataset(dir_root):

    file1 = os.listdir(dir_root)
    num_class = len(file1)
    total_images = 0
    # dict_num_images = {}
    for i in range(num_class):
        path_temp = os.path.join(dir_root, file1[i])
        file2 = os.listdir(path_temp)
        num_images_one_class = len(file2)
        total_images = total_images + num_images_one_class
        # dict_num_images[file1[i]] = num_images_one_class

    return num_class, total_images


def get_deep_model(dir_model, num_class):

    device = torch.device('cpu')

    model = models.resnet18()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_class)
    checkpoint_shape = torch.load(dir_model, map_location=device)
    model.load_state_dict(checkpoint_shape['state_dict'])

    return model


def get_deep_results(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)
    soft_output = F.softmax(outputs, dim=1)

    return soft_output


def main():
    num_class, total_images = get_info_of_dataset(os.path.join(dir_root, 'shape', dataset_type))

    print('The dataset name is: ')
    print(dataset_name)
    print('\n')
    print('The dataset type is: ')
    print(dataset_type)
    print('\n')
    print('The total number of images is: ')
    print(total_images)
    print('\n')

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_original = get_deep_model(dir_model_original, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()
    model_original.eval()

    num_correct_shape = torch.zeros(num_class)
    num_total_shape = torch.zeros(num_class)
    num_correct_color = torch.zeros(num_class)
    num_total_color = torch.zeros(num_class)
    num_correct_texture = torch.zeros(num_class)
    num_total_texture = torch.zeros(num_class)
    num_correct_original = torch.zeros(num_class)
    num_total_original = torch.zeros(num_class)

    val_loader = get_Dataloader(dir_dataset_shape, dir_dataset_texture, dir_dataset_color, dir_dataset_original, total_images)

    for idx, (texture_img, shape_img, color_img, original_img, label, img_name) in enumerate(val_loader):
        # Shape
        output_shape = get_deep_results(shape_img, model_shape)
        _, shape_pre = torch.max(output_shape, dim=1)

        for j in range(total_images):
            if label[j] == shape_pre[j]:
                num_correct_shape[label[j]] = num_correct_shape[label[j]] + 1
                num_total_shape[label[j]] = num_total_shape[label[j]] + 1
            else:
                num_total_shape[label[j]] = num_total_shape[label[j]] + 1

        correct_shape = (label == shape_pre).sum()
        shape_acc = correct_shape / total_images
        print('shape_accuracy:')
        print(shape_acc)
        # f1_shape = f1_score(label, shape_pre, average='micro')
        f1_shape = f1_score(label, shape_pre, average='macro')
        print('shape_f1_score:')
        print('%.4f'%f1_shape)
        print()
        print()


        #color
        output_color = get_deep_results(color_img, model_color)
        _, color_pre = torch.max(output_color, dim=1)

        for j in range(total_images):
            if label[j] == color_pre[j]:
                num_correct_color[label[j]] = num_correct_color[label[j]] + 1
                num_total_color[label[j]] = num_total_color[label[j]] + 1
            else:
                num_total_color[label[j]] = num_total_color[label[j]] + 1

        correct_color = (label == color_pre).sum()
        color_acc = correct_color / total_images
        print('color:')
        print(color_acc)
        # f1_color = f1_score(label, color_pre, average='micro')
        f1_color = f1_score(label, color_pre, average='macro')
        print('color_f1_score:')
        print('%.4f'%f1_color)
        print()
        print()


        # texture
        output_texture = get_deep_results(texture_img, model_texture)
        _, texture_pre = torch.max(output_texture, dim=1)

        for j in range(total_images):
            if label[j] == texture_pre[j]:
                num_correct_texture[label[j]] = num_correct_texture[label[j]] + 1
                num_total_texture[label[j]] = num_total_texture[label[j]] + 1
            else:
                num_total_texture[label[j]] = num_total_texture[label[j]] + 1

        correct_texture = (label == texture_pre).sum()
        texture_acc = correct_texture / total_images
        print('texture:')
        print(texture_acc)
        # f1_texture = f1_score(label, texture_pre, average='micro')
        f1_texture = f1_score(label, texture_pre, average='macro')
        print('texture_f1_score:')
        print('%.4f'%f1_texture)





    print()






if __name__ == '__main__':
    # Parameters
    # dataset_name = 'color_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'val'
    
    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '45.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '12.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '38.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '29.pth')
    
    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    # ————————————————————————————————————————————————————————————————————————————————————————————————————
    # dataset_name = 'shape_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'val'
    
    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '12.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '20.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '15.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '34.pth')
    
    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    # ————————————————————————————————————————————————————————————————————————————————————————————————————
    # dataset_name = 'texture_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'val'

    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '33.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '43.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '34.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '39.pth')

    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    #-----------------------------------------------------------------------------------------------
    dataset_name = 'all_datasets'
    dir_root = os.path.join('data', dataset_name, 'feature_images')
    dataset_type = 'val'
    
    dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '8.pth')
    dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '16.pth')
    dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '23.pth')
    dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '25.pth')
    
    dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)


    main()