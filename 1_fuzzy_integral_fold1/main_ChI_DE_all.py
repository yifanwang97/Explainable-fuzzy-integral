import numpy as np
import geatpy as ea
import os
import torch.nn.functional as F
from data_loader import get_Dataloader
import torch
from torch import nn
import torchvision.models as models
import sympy as sp


class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, output_deepnets_train, label_train):
        self.output_deepnets_train = output_deepnets_train
        self.label_train = label_train
        num_integral = len(output_deepnets_train)
        num_class = output_deepnets_train['output_color'].shape[1]
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = num_integral * num_class  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = np.zeros(Dim)  # 决策变量下界
        ub = np.ones(Dim)  # 决策变量上界
        lbin = np.zeros(Dim)  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = np.zeros(Dim)  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars): # 目标函数
        label_train = self.label_train
        num_images = label_train.shape
        [population_size, num_param] = Vars.shape
        num_class = int(num_param / 3)
        ACC_list = []
        CV_list = []
        for i_population in range(population_size):
            # _______________________________________________________________________
            # Computing the accuracy with each population
            num_CI_correct = torch.tensor([0])
            g_group_temp0 = Vars[i_population, :]
            g_group_list_temp = np.array_split(g_group_temp0, num_class)
            # Computing the fuzzy integral results
            choquet_integral_results = self.computing_choquet_integral_for_training(g_group_list_temp)
            _, CI_pre = torch.max(choquet_integral_results, dim=1)
            CI_correct_temp = (label_train == CI_pre).sum()
            num_CI_correct = num_CI_correct + CI_correct_temp
            ACC_single_population = (num_CI_correct / torch.tensor(num_images)).numpy()
            ACC_list.append(ACC_single_population)

        ACC_all = np.array(ACC_list)
        return ACC_all


    def computing_choquet_integral_for_training(self, gi_each_class):
        output_deepnets = self.output_deepnets_train
        num_integral = len(output_deepnets)
        [num_images, num_class] = output_deepnets['output_color'].shape
        FI_results = torch.zeros([num_images, num_class], dtype=torch.float)
        g_matrix = torch.zeros([num_images, num_integral], dtype=torch.float)

        for i_class in range(num_class):
            # Compute g1, g2, g3, g12, g23, g13, g123
            g_group_pre = gi_each_class[i_class]
            g_group = self.computing_g_for_training(g_group_pre)

            integral_matrix = torch.cat(
                (output_deepnets['output_shape'][:, i_class].unsqueeze(1),
                 output_deepnets['output_texture'][:, i_class].unsqueeze(1),
                 output_deepnets['output_color'][:, i_class].unsqueeze(1)), dim=1)
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
            FI_results[:, i_class] = FI_results_one_class

        return FI_results

    def computing_g_for_training(self, g_group_pre):
        g1 = g_group_pre[0]
        g2 = g_group_pre[1]
        g3 = g_group_pre[2]
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

def computing_choquet_integral_for_testing(num_images, num_class, output_deepnets, label_list, gi_each_class):
    num_integral = len(output_deepnets)
    FI_results = torch.zeros([num_images, num_class], dtype=torch.float)
    g_matrix = torch.zeros([num_images, num_integral], dtype=torch.float)

    for i_class in range(num_class):
        # Compute g1, g2, g3, g12, g23, g13, g123
        class_name = label_list[i_class]
        g_group = gi_each_class[class_name]

        integral_matrix = torch.cat(
            (output_deepnets['output_shape'][:, i_class].unsqueeze(1),
             output_deepnets['output_texture'][:, i_class].unsqueeze(1),
             output_deepnets['output_color'][:, i_class].unsqueeze(1)), dim=1)
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
        FI_results[:, i_class] = FI_results_one_class

    return FI_results

def computing_all_g(g_group_pre):
    g1 = g_group_pre[0]
    g2 = g_group_pre[1]
    g3 = g_group_pre[2]
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

def main():

    # For val dataset
    num_class, total_images_val = get_info_of_dataset(os.path.join(dir_root, 'shape', 'val'))

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()

    batch_size_val = total_images_val
    val_loader = get_Dataloader(dir_dataset_shape_val, dir_dataset_texture_val, dir_dataset_color_val,
                                dir_dataset_original_val, batch_size_val)

    for idx, (texture_img_val, shape_img_val, color_img_val, original_img_val, label_val, img_name_val) in enumerate(val_loader):
        output_shape = get_deep_results(shape_img_val, model_shape)
        output_texture = get_deep_results(texture_img_val, model_texture)
        output_color = get_deep_results(color_img_val, model_color)

        output_deepnets_val = {'output_color': output_color, 'output_shape': output_shape,
                                 'output_texture': output_texture}

        # The names of classes and their corresponding labels
        label_list = val_loader.dataset.label_list
        problem = MyProblem(output_deepnets_val, label_val)
        algorithm = ea.soea_DE_rand_1_bin_templet(problem, ea.Population(Encoding='RI', NIND=50), MAXGEN=200,
                                                  # 最大进化代数。
                                                  logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
        algorithm.mutOper.F = 0.5  # 差分进化中的参数F
        algorithm.recOper.XOVR = 0.7  # 重组概率
        # 求解
        res = ea.optimize(algorithm,
                          verbose=True,
                          drawing=1,
                          outputMsg=True,
                          drawLog=False,
                          saveFlag=True)
        print(res)
    param_g_all = np.squeeze(res['Vars'])
    g_pre_array = np.array_split(param_g_all, num_class)
    gi_each_class = {}
    for i_class in range(num_class):
        g_temp_array = g_pre_array[i_class]
        gi_each_class_temp = computing_all_g(g_temp_array)
        gi_each_class[label_list[i_class]] = gi_each_class_temp
    print()
    print()
    print('The gi_each_class is:')
    print(gi_each_class)


    # For testing dataset
    _, total_images_test = get_info_of_dataset(os.path.join(dir_root, 'shape', 'test'))
    batch_size_test = total_images_test
    test_loader = get_Dataloader(dir_dataset_shape_test, dir_dataset_texture_test, dir_dataset_color_test,
                                 dir_dataset_original_test, batch_size_test)

    num_CI_correct = torch.tensor([0])

    for idx, (texture_img_test, shape_img_test, color_img_test, original_img_test, label_test, img_name) in enumerate(test_loader):
        output_shape = get_deep_results(shape_img_test, model_shape)
        output_texture = get_deep_results(texture_img_test, model_texture)
        output_color = get_deep_results(color_img_test, model_color)

        output_deepnets_test = {'output_color': output_color, 'output_shape': output_shape,
                                'output_texture': output_texture}

        # The names of classes and their corresponding labels
        label_list = test_loader.dataset.label_list

        choquet_integral_results = computing_choquet_integral_for_testing(batch_size_test, num_class,
                                                                          output_deepnets_test, label_list, gi_each_class)

        _, CI_pre = torch.max(choquet_integral_results, dim=1)
        CI_correct_temp = (label_test == CI_pre).sum()
        num_CI_correct = num_CI_correct + CI_correct_temp

    ACC_CI = num_CI_correct / total_images_test

    print('\n')
    print('\n')
    print('The dataset name is: ')
    print(dataset_name)
    print('\n')
    print('The total number of images is: ')
    print(total_images_test)
    print('\n')
    print('Choquet integral accuracy is: ')
    print(ACC_CI.numpy())
    print('\n')
    print()





if __name__ == '__main__':
    # Parameters
    dataset_name = 'all_datasets'
    dir_root = os.path.join('data', dataset_name, 'feature_images')

    dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '8.pth')
    dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '16.pth')
    dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '23.pth')

    dir_dataset_color_val = os.path.join(dir_root, 'color', 'val')
    dir_dataset_shape_val = os.path.join(dir_root, 'shape', 'val')
    dir_dataset_texture_val = os.path.join(dir_root, 'texture', 'val')
    dir_dataset_original_val = os.path.join('data', dataset_name, 'original_images', 'val')

    dir_dataset_color_test = os.path.join(dir_root, 'color', 'test')
    dir_dataset_shape_test = os.path.join(dir_root, 'shape', 'test')
    dir_dataset_texture_test = os.path.join(dir_root, 'texture', 'test')
    dir_dataset_original_test = os.path.join('data', dataset_name, 'original_images', 'test')

    main()
