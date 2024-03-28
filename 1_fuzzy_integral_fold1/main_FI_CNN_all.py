import sympy as sp
import torchvision.models as models
import torch
from torch import nn
# import torchvision.datasets as datasets
import os
# import torchvision.transforms as transforms
import torch.nn.functional as F
from data_loader import get_Dataloader
import copy


def compute_fuzzy_measure(g_input):
    # g1 is shape, g2 is texture, and g3 is color.
    g1 = g_input[0]
    g2 = g_input[1]
    g3 = g_input[2]
    lambda_ori = sp.Symbol('x')
    f_original = (1 + lambda_ori * g1) * (1 + lambda_ori * g2) * (1 + lambda_ori * g3) - (lambda_ori + 1)
    lambda_ori = sp.Symbol('x')
    f = sp.expand(f_original)
    lambda_ori = sp.solve(f)
    lambda_ori.remove(0)
    g_lambda = lambda_ori[1]
    g12 = g1 + g2 + g_lambda * g1 * g2
    g13 = g1 + g3 + g_lambda * g1 * g3
    g23 = g2 + g3 + g_lambda * g2 * g3

    g_group = {
        'g1': g1,
        'g2': g2,
        'g3': g3,
        'g12': g12,
        'g13': g13,
        'g23': g23,
        'g123': 1.0
    }

    return g_group


def get_info_of_dataset(dir_root):

    file1 = os.listdir(dir_root)
    num_class = len(file1)
    num_images = []
    total_images = 0
    for i in range(num_class):
        path_temp = os.path.join(dir_root, file1[i])
        file2 = os.listdir(path_temp)
        num_images.append(len(file2))
        total_images = total_images + num_images[i]

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

def choquet_integral(num_images, num_class, output_deepnets, g_group):
    num_integral = 3
    FI_results = torch.zeros([num_images, num_class], dtype=torch.float)
    g_matrix = torch.zeros([num_images, num_integral], dtype=torch.float)

    for i in range(num_class):
        integral_matrix = torch.cat(
            (output_deepnets['output_shape'][:, i].unsqueeze(1),
             output_deepnets['output_texture'][:, i].unsqueeze(1),
             output_deepnets['output_color'][:, i].unsqueeze(1)), dim=1)
        sorted_output, sorted_indices = torch.sort(integral_matrix, descending=True, dim=-1)

        for j in range(num_images):
            if sorted_indices[j, :].equal(torch.tensor([0, 1, 2])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g1'], g_group['g12'] - g_group['g1'], g_group['g123'] - g_group['g12']],
                    dtype=torch.float)
            elif sorted_indices[j, :].equal(torch.tensor([0, 2, 1])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g1'], g_group['g13'] - g_group['g1'], g_group['g123'] - g_group['g13']],
                    dtype=torch.float)
            elif sorted_indices[j, :].equal(torch.tensor([1, 0, 2])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g2'], g_group['g12'] - g_group['g2'], g_group['g123'] - g_group['g12']],
                    dtype=torch.float)
            elif sorted_indices[j, :].equal(torch.tensor([1, 2, 0])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g2'], g_group['g23'] - g_group['g2'], g_group['g123'] - g_group['g23']],
                    dtype=torch.float)
            elif sorted_indices[j, :].equal(torch.tensor([2, 0, 1])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g3'], g_group['g13'] - g_group['g3'], g_group['g123'] - g_group['g13']],
                    dtype=torch.float)
            elif sorted_indices[j, :].equal(torch.tensor([2, 1, 0])):
                g_matrix[j, :] = torch.tensor(
                    [g_group['g3'], g_group['g23'] - g_group['g3'], g_group['g123'] - g_group['g23']],
                    dtype=torch.float)
            else:
                print('A mistake in Choquet integral, but you are so great.')

        fuzzy_integral = sorted_output * g_matrix
        FI_results_one_class = torch.sum(fuzzy_integral, dim=1)
        FI_results[:, i] = FI_results_one_class

    return FI_results

def computing_g_initial(accuracy_rate, f1_score):
    min_accuracy = min(accuracy_rate)

    g_initial = [0, 0, 0]
    g_initial_pre = [(accuracy_rate[i]+f1_score[i])/2-min_accuracy for i in range(3)]

    for i in range(len(g_initial_pre)):
        if g_initial_pre[i] < 0:
            g_initial[i] = 0.01
        else:
            g_initial[i] = round(abs(g_initial_pre[i]), 2)
    
    return g_initial

def main():

    num_class, total_images = get_info_of_dataset(os.path.join(dir_root, 'shape', dataset_type))

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()

    # single feature
    ACC_CI_list = [] 
    # all features
    ACC_CI_list_all = []
    g_initial = computing_g_initial(accuracy_rate, f1_score)
    print()
    print()
    print('g_initial:')
    print(g_initial)
    g_input = copy.deepcopy(g_initial)
    # adjust gi 
    g_bias_normal = [-0.02, -0.01, 0, 0.01, 0.02]
    g_bias_min = [0, 0.01, 0.02, 0.03, 0.04]
    i_count = 0

    for i in range(3):
        if g_initial[i] <= 0.02:
            for g_bias in g_bias_min:
                g_input[i] = g_initial[i] + g_bias
                g_group = compute_fuzzy_measure(g_input)

                num_CI_correct = torch.tensor([0])

                val_loader = get_Dataloader(dir_dataset_shape, dir_dataset_texture, dir_dataset_color, dir_dataset_original, total_images)

                for idx, (texture_img, shape_img, color_img, original_img, label, img_name) in enumerate(val_loader):
                    output_shape = get_deep_results(shape_img, model_shape)
                    output_texture = get_deep_results(texture_img, model_texture)
                    output_color = get_deep_results(color_img, model_color)

                    output_deepnets = {'output_color': output_color, 'output_shape': output_shape, 'output_texture': output_texture}

                    choquet_integral_results = choquet_integral(total_images, num_class, output_deepnets, g_group)

                    _, CI_pre = torch.max(choquet_integral_results, dim=1)
                    CI_correct_temp = (label == CI_pre).sum()
                    num_CI_correct = num_CI_correct + CI_correct_temp
                    
                ACC_CI = num_CI_correct / total_images
                ACC_CI_list.append(ACC_CI)
                ACC_CI_list_all.append(ACC_CI)
                
                print()
                print()
                print(i_count)
                print('g_measure:')
                print(g_input)
                print('\n')
                print('Choquet integral accuracy is: ')
                print(ACC_CI.numpy())
                print('\n')
                i_count = i_count + 1
            
            best_ACC_single_feature = max(ACC_CI_list)
            best_index_single_feature = ACC_CI_list.index(best_ACC_single_feature)
            g_input[i] = g_initial[i] + g_bias_min[best_index_single_feature]
            ACC_CI_list.clear()
                   
        else:
            for g_bias in g_bias_normal:
                g_input[i] = g_initial[i] + g_bias
                g_group = compute_fuzzy_measure(g_input)

                num_CI_correct = torch.tensor([0])

                val_loader = get_Dataloader(dir_dataset_shape, dir_dataset_texture, dir_dataset_color, dir_dataset_original, total_images)

                for idx, (texture_img, shape_img, color_img, original_img, label, img_name) in enumerate(val_loader):
                    output_shape = get_deep_results(shape_img, model_shape)
                    output_texture = get_deep_results(texture_img, model_texture)
                    output_color = get_deep_results(color_img, model_color)

                    output_deepnets = {'output_color': output_color, 'output_shape': output_shape, 'output_texture': output_texture}

                    choquet_integral_results = choquet_integral(total_images, num_class, output_deepnets, g_group)

                    _, CI_pre = torch.max(choquet_integral_results, dim=1)
                    CI_correct_temp = (label == CI_pre).sum()
                    num_CI_correct = num_CI_correct + CI_correct_temp
                    
                ACC_CI = num_CI_correct / total_images
                ACC_CI_list.append(ACC_CI)
                ACC_CI_list_all.append(ACC_CI)
                
                print()
                print()
                print(i_count)
                print('g_measure:')
                print(g_input)
                print('\n')
                print('Choquet integral accuracy is: ')
                print(ACC_CI.numpy())
                print('\n')
                i_count = i_count + 1

            best_ACC_single_feature = max(ACC_CI_list)
            best_index_single_feature = ACC_CI_list.index(best_ACC_single_feature)
            g_input[i] = g_initial[i] + g_bias_normal[best_index_single_feature]
            ACC_CI_list.clear()

        
    best_ACC = max(ACC_CI_list_all)
    best_index = ACC_CI_list_all.index(best_ACC)
    print('The dataset name is: ')
    print(dataset_name)
    print('\n')
    print('The dataset type is: ')
    print(dataset_type)
    print('\n')
    print('The total number of images is: ')
    print(total_images)
    print('\n')
    print('Best Choquet integral accuracy is: ')
    print(best_ACC.numpy())
    print('\n')
    print('Best index is: ')
    print(best_index)
    print('\n')




if __name__ == '__main__':
    # Parameters
    # dataset_name = 'color_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'test'
    
    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '45.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '12.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '38.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '29.pth')
    
    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    # accuracy_rate = [0.68, 0.76, 0.82]
    # f1_score = [0.65, 0.77, 0.81]

    # ————————————————————————————————————————————————————————————————————————————————————————————————————
    # dataset_name = 'shape_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'test'
    
    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '12.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '20.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '15.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '34.pth')
    
    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    # accuracy_rate = [0.90, 0.82, 0.68]
    # f1_score = [0.89, 0.82, 0.67]

    # ————————————————————————————————————————————————————————————————————————————————————————————————————
    # dataset_name = 'texture_biased_dataset'
    # dir_root = os.path.join('data', dataset_name, 'feature_images')
    # dataset_type = 'test'

    # dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '33.pth')
    # dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '43.pth')
    # dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '34.pth')
    # dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '39.pth')

    # dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    # dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    # dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    # dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    # accuracy_rate = [0.61, 0.81, 0.66]
    # f1_score = [0.59, 0.80, 0.65]

    #-----------------------------------------------------------------------------------------------
    dataset_name = 'all_datasets'
    dir_root = os.path.join('data', dataset_name, 'feature_images')
    dataset_type = 'test'
    
    dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '8.pth')
    dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '16.pth')
    dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '23.pth')
    dir_model_original = os.path.join('data', dataset_name, 'model/original_resnet18', '25.pth')
    
    dir_dataset_color = os.path.join(dir_root, 'color', dataset_type)
    dir_dataset_shape = os.path.join(dir_root, 'shape', dataset_type)
    dir_dataset_texture = os.path.join(dir_root, 'texture', dataset_type)
    dir_dataset_original = os.path.join('data', dataset_name, 'original_images', dataset_type)

    accuracy_rate = [0.70, 0.76, 0.67]
    f1_score = [0.69, 0.76, 0.65]

    main()