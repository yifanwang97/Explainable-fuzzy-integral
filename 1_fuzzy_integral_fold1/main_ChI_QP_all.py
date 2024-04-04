import numpy as np
import itertools
from cvxopt import solvers, matrix
import torchvision.models as models
import torch
from torch import nn
# import torchvision.datasets as datasets
import os
# import torchvision.transforms as transforms
import torch.nn.functional as F
from data_loader import get_Dataloader


class ChoquetIntegral:

    def __init__(self):
        """Instantiation of a ChoquetIntegral.

           This sets up the ChI. It doesn't take any input parameters
           because you may want to use pass your own values in(as opposed
           to learning from data). To instatiate, use
           chi = ChoquetIntegral.ChoquetIntegral()
        """
        self.trainSamples, self.trainLabels = [], []
        self.testSamples, self.testLabels = [], []
        self.N, self.numberConstraints, self.M = 0, 0, 0
        self.g = 0
        self.fm = []
        self.type = []


    def train_chi(self, x1, l1):
        """
        This trains this instance of your ChoquetIntegral w.r.t x1 and l1.

        :param x1: These are the training samples of size N x M(inputs x number of samples)
        :param l1: These are the training labels of size 1 x M(label per sample)

        """
        self.type = 'quad'
        self.trainSamples = x1
        self.trainLabels = l1
        self.N = self.trainSamples.shape[0]
        self.M = self.trainSamples.shape[1]
        print("Number Inputs : ", self.N, "; Number Samples : ", self.M)
        self.fm = self.produce_lattice()

        return self



    def chi_quad(self, x2):
        """
        This will produce an output for this instance of the ChI

        This will use the learned(or specified) Choquet integral to
        produce an output w.r.t. to the new input.

        :param x2: testing sample
        :return: output of the choquet integral.
        """
        if self.type == 'quad':
            n = len(x2)
            pi_i = np.argsort(x2)[::-1][:n] + 1
            ch = x2[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
            for i in range(1, n):
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                ch = ch + x2[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
            return ch
        else:
            print("If using sugeno measure, you need to use chi_sugeno.")


    def produce_lattice(self):
        """
            This method builds is where the lattice(or FM variables) will be learned.

            The FM values can be found via a quadratic program, which is used here
            after setting up constraint matrices. Refer to papers for complete overview.

        :return: Lattice, the learned FM variables.
        """

        fm_len = 2 ** self.N - 1  # nc
        E = np.zeros((fm_len, fm_len))  # D
        L = np.zeros(fm_len)  # f
        index_keys = self.get_keys_index()
        for i in range(0, self.M):  # it's going through one sample at a time.
            l = self.trainLabels[i]  # this is the labels
            fm_coeff = self.get_fm_class_img_coeff(index_keys, self.trainSamples[:, i], fm_len)  # this is Hdiff
            # print(fm_coeff)
            L = L + (-2) * l * fm_coeff
            E = E + np.matmul(fm_coeff.reshape((fm_len, 1)), fm_coeff.reshape((1, fm_len)))

        G, h, A, b = self.build_constraint_matrices(index_keys, fm_len)
        sol = solvers.qp(matrix(2 * E, tc='d'), matrix(L.T, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'),
                         matrix(A, tc='d'), matrix(b, tc='d'))

        g = sol['x']
        Lattice = {}
        for key in index_keys.keys():
            Lattice[key] = g[index_keys[key]]
        return Lattice


    def build_constraint_matrices(self, index_keys, fm_len):
        """
        This method builds the necessary constraint matrices.

        :param index_keys: map to reference lattice components
        :param fm_len: length of the fuzzy measure
        :return: the constraint matrices
        """

        vls = np.arange(1, self.N + 1)
        line = np.zeros(fm_len)
        G = line
        line[index_keys[str(np.array([1]))]] = -1.
        h = np.array([0])
        for i in range(2, self.N + 1):
            line = np.zeros(fm_len)
            line[index_keys[str(np.array([i]))]] = -1.
            G = np.vstack((G, line))
            h = np.vstack((h, np.array([0])))
        for i in range(2, self.N + 1):
            parent = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in parent:
                for j in range(len(latt_pt) - 1, len(latt_pt)):
                    children = np.array(list(itertools.combinations(latt_pt, j)))
                    for latt_ch in children:
                        line = np.zeros(fm_len)
                        line[index_keys[str(latt_ch)]] = 1.
                        line[index_keys[str(latt_pt)]] = -1.
                        G = np.vstack((G, line))
                        h = np.vstack((h, np.array([0])))

        line = np.zeros(fm_len)
        line[index_keys[str(vls)]] = 1.
        G = np.vstack((G, line))
        h = np.vstack((h, np.array([1])))

        # equality constraints
        A = np.zeros((1, fm_len))
        A[0, -1] = 1
        b = np.array([1]);

        return G, h, A, b


    def get_fm_class_img_coeff(self, Lattice, h, fm_len):  # Lattice is FM_name_and_index, h is the samples, fm_len
        """
        This creates a FM map with the name as the key and the index as the value

        :param Lattice: dictionary with FM
        :param h: sample
        :param fm_len: fm length
        :return: the fm_coeff
        """

        n = len(h)  # len(h) is the number of the samples
        fm_coeff = np.zeros(fm_len)
        pi_i = np.argsort(h)[::-1][:n] + 1
        for i in range(1, n):
            fm_coeff[Lattice[str(np.sort(pi_i[:i]))]] = h[pi_i[i - 1] - 1] - h[pi_i[i] - 1]
        fm_coeff[Lattice[str(np.sort(pi_i[:n]))]] = h[pi_i[n - 1] - 1]
        np.matmul(fm_coeff, np.transpose(fm_coeff))
        return fm_coeff


    def get_keys_index(self):
        """
        Sets up a dictionary for referencing FM.

        :return: The keys to the dictionary
        """

        vls = np.arange(1, self.N + 1)
        count = 0
        Lattice = {}
        for i in range(0, self.N):
            Lattice[str(np.array([vls[i]]))] = count
            count = count + 1
        for i in range(2, self.N + 1):
            A = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in A:
                Lattice[str(latt_pt)] = count
                count = count + 1
        return Lattice

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



def main():

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

        # The names of classes and their corresponding labels
        label_list = val_loader.dataset.label_list

        gi_each_class = {}
        for i_class in range(len(label_list)):
            training_data_each_class = torch.cat(
                (output_shape[:, i_class].unsqueeze(1), output_texture[:, i_class].unsqueeze(1), output_color[:, i_class].unsqueeze(1)), dim=1)
            training_label_each_class = torch.zeros(len(label_val))
            for i_patterns in range(len(label_val)):
                if label_val[i_patterns] == i_class:
                    training_label_each_class[i_patterns] = 1
            training_data_temp0 = training_data_each_class.numpy()
            training_data_temp1 = np.transpose(training_data_temp0)
            training_label_temp = training_label_each_class.numpy()
            chi = ChoquetIntegral()
            chi.train_chi(training_data_temp1, training_label_temp)
            print('class_name: ')
            print(label_list[i_class])
            print(chi.fm)
            gi_each_class_temp = {'g1': chi.fm['[1]'],
                                  'g2': chi.fm['[2]'],
                                  'g3': chi.fm['[3]'],
                                  'g12': chi.fm['[1 2]'],
                                  'g13': chi.fm['[1 3]'],
                                  'g23': chi.fm['[2 3]'],
                                  'g123': chi.fm['[1 2 3]']
                                  }
            gi_each_class[label_list[i_class]] = gi_each_class_temp

    # For test dataset

    _, total_images_test = get_info_of_dataset(os.path.join(dir_root, 'shape', 'test'))
    batch_size_test = total_images_test
    test_loader = get_Dataloader(dir_dataset_shape_test, dir_dataset_texture_test, dir_dataset_color_test, dir_dataset_original_test, batch_size_test)

    num_CI_correct = torch.tensor([0])

    for idx, (texture_img_test, shape_img_test, color_img_test, original_img_test, label_test, img_name) in enumerate(test_loader):

        output_shape = get_deep_results(shape_img_test, model_shape)
        output_texture = get_deep_results(texture_img_test, model_texture)
        output_color = get_deep_results(color_img_test, model_color)

        output_deepnets_test = {'output_color': output_color, 'output_shape': output_shape, 'output_texture': output_texture}

        # The names of classes and their corresponding labels
        label_list = test_loader.dataset.label_list

        choquet_integral_results = computing_choquet_integral(batch_size_test, num_class, output_deepnets_test, label_list, gi_each_class)

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





