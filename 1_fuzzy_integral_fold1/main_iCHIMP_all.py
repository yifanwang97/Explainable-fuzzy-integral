import numpy as np
import itertools
from cvxopt import solvers, matrix
import torchvision.models as models
import torch
# from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as datasets
import os
# import torchvision.transforms as transforms
import torch.nn.functional as F
from data_loader import get_Dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}" + str(N) + "b}"
    a = []
    for i in range(1, 2 ** N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))

    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2 ** (i) for i in sourcesInNodes]

    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s)
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]

    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j], '1')
        sourcesInNode.append(idxLR)
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j, idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]


class Choquet_integral(torch.nn.Module):

    def __init__(self, N_in, N_out):
        super(Choquet_integral, self).__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2 ** self.N_in - 2

        # The FM is initialized with mean
        dummy = (1. / self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
        #        self.vars = torch.nn.Parameter( torch.Tensor(self.nVars,N_out))
        self.vars = torch.nn.Parameter(dummy)

        # following function uses numpy vs pytorch
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.N_in)

        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]

    def forward(self, inputs):
        self.FM = self.chi_nn_vars(self.vars)
        sortInputs, sortInd = torch.sort(inputs, 1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M, 1).cuda()), 1)
        sortInputs = sortInputs[:, :-1] - sortInputs[:, 1:]

        out = torch.cumsum(torch.pow(2, sortInd), 1) - torch.ones(1, dtype=torch.int64).cuda()

        data = torch.zeros((M, self.nVars + 1)).cuda()

        for i in range(M):
            data[i, out[i, :]] = sortInputs[i, :]

        ChI = torch.matmul(data, self.FM)

        return ChI

    # Converts NN-vars to FM vars
    def chi_nn_vars(self, chi_vars):
        #        nVars,_ = chi_vars.size()
        chi_vars = torch.abs(chi_vars)
        #        nInputs = inputs.get_shape().as_list()[1]

        FM = chi_vars[None, 0, :]
        for i in range(1, self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM, chi_vars[None, i, :]), 0)
            else:
                #         ss=tf.gather_nd(variables, [[1],[2]])
                maxVal, _ = torch.max(FM[indices, :], 0)
                temp = torch.add(maxVal, chi_vars[i, :])
                FM = torch.cat((FM, temp[None, :]), 0)

        FM = torch.cat([FM, torch.ones((1, self.N_out)).cuda()], 0)
        FM = torch.min(FM, torch.ones(1).cuda())

        return FM

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


def computing_choquet_integral(num_images, num_class, output_deepnets, label_list, gi_each_class):
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


def training_iCHIMP(training_patterns, training_labels, net, criterion, optimizer):
    dataset = torch.utils.data.TensorDataset(training_patterns, training_labels)
    data_training_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    for iter, (training_patterns, training_labels) in enumerate(data_training_loader):
        output = net(training_patterns)
        loss = criterion(output, training_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def computing_accuracy(dataset_type, gi_each_class):

    num_class, total_images = get_info_of_dataset(os.path.join(dir_root, 'shape', dataset_type))

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()

    batch_size = total_images
    if dataset_type == 'val':
        dir_dataset_color = dir_dataset_color_val
        dir_dataset_shape = dir_dataset_shape_val
        dir_dataset_texture = dir_dataset_texture_val
        dir_dataset_original = dir_dataset_original_val
    elif dataset_type == 'test':
        dir_dataset_color = dir_dataset_color_test
        dir_dataset_shape = dir_dataset_shape_test
        dir_dataset_texture = dir_dataset_texture_test
        dir_dataset_original = dir_dataset_original_test

    data_loader = get_Dataloader(dir_dataset_shape, dir_dataset_texture, dir_dataset_color, dir_dataset_original, batch_size)

    num_CI_correct = torch.tensor([0])

    for idx, (texture_img, shape_img, color_img, original_img, label, img_name) in enumerate(data_loader):

        output_shape = get_deep_results(shape_img, model_shape)
        output_texture = get_deep_results(texture_img, model_texture)
        output_color = get_deep_results(color_img, model_color)

        output_deepnets = {'output_color': output_color, 'output_shape': output_shape, 'output_texture': output_texture}

        # The names of classes and their corresponding labels
        label_list = data_loader.dataset.label_list

        choquet_integral_results = computing_choquet_integral(batch_size, num_class, output_deepnets, label_list, gi_each_class)

        _, CI_pre = torch.max(choquet_integral_results, dim=1)
        CI_correct_temp = (label == CI_pre).sum()
        num_CI_correct = num_CI_correct + CI_correct_temp

    ACC_CI = num_CI_correct / total_images

    print('dataset type is:')
    print(dataset_type)
    print('Choquet integral accuracy is: ')
    print(ACC_CI.numpy())
    return ACC_CI, total_images



def main():

    # 1 Get the inputs and labels of validation datasets for Choquet Integral neural network
    num_class, total_images_val = get_info_of_dataset(os.path.join(dir_root, 'shape', 'val'))

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()

    batch_size_val = total_images_val
    val_loader = get_Dataloader(dir_dataset_shape_val, dir_dataset_texture_val, dir_dataset_color_val, dir_dataset_original_val, batch_size_val)

    for idx, (texture_img_val, shape_img_val, color_img_val, original_img_val, label_val, img_name_val) in enumerate(val_loader):

        output_shape = get_deep_results(shape_img_val, model_shape)
        output_texture = get_deep_results(texture_img_val, model_texture)
        output_color = get_deep_results(color_img_val, model_color)

        label_list = val_loader.dataset.label_list

        training_patterns_list = []
        for i_class in range(num_class):
            training_patterns_each_class = torch.cat(
                (output_shape[:, i_class].unsqueeze(1), output_texture[:, i_class].unsqueeze(1), output_color[:, i_class].unsqueeze(1)), dim=1)
            training_patterns_list.append(training_patterns_each_class)

        training_patterns = torch.cat(training_patterns_list, 0)

        training_labels_one_dim = label_val.repeat(num_class)
        training_labels = F.one_hot(training_labels_one_dim, num_class).float()

        training_patterns = training_patterns.cuda()
        training_labels = training_labels.cuda()
        
    # 2 Define the model, optimizer, and loss function for Choquet Integral neural network
    num_of_sources = training_patterns.shape[1]
    net = Choquet_integral(num_of_sources, num_class)
    net = net.cuda()

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum, weight_decay)

    # 3 Training the Choquet Integral neural network
    best_epoch = 0
    best_val_accuracy = 0
    best_gi_each_class = {}
    corresponding_test_accuracy = 0

    for i_epoch in range(num_epochs):
        training_iCHIMP(training_patterns, training_labels, net, criterion, optimizer)
        FM_learned_temp = (net.chi_nn_vars(net.vars).cpu()).detach().numpy()
        gi_each_class = {}
        for i_class in range(num_class):
            gi_each_class_temp = {'g1': FM_learned_temp[0, i_class],
                                    'g2': FM_learned_temp[1, i_class],
                                    'g3': FM_learned_temp[3, i_class],
                                    'g12': FM_learned_temp[2, i_class],
                                    'g13': FM_learned_temp[4, i_class],
                                    'g23': FM_learned_temp[5, i_class],
                                    'g123': FM_learned_temp[6, i_class]
                                    }
            gi_each_class[label_list[i_class]] = gi_each_class_temp
        print('The number of epoches:')
        print(i_epoch)
        print('gi for each class is:')
        print(gi_each_class)

        val_accuracy_temp, _ = computing_accuracy('val', gi_each_class)
        test_accuracy_temp, total_images = computing_accuracy('test', gi_each_class)
        print('\n')

        if val_accuracy_temp > best_val_accuracy:
            best_val_accuracy = val_accuracy_temp
            corresponding_test_accuracy = test_accuracy_temp
            best_epoch = i_epoch
            best_gi_each_class = gi_each_class
    print('The dataset name is: ')
    print(dataset_name)
    print('The number of total images is:')
    print(total_images)
    print('best_epoch is:')
    print(best_epoch)
    print('best_val_accuracy is:')
    print(best_val_accuracy)
    print('best_gi_each_class is:')
    print(best_gi_each_class)
    print('corresponding_test_accuracy is:')
    print(corresponding_test_accuracy)




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

    learning_rate = 0.001
    num_epochs = 100
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128

    main()





