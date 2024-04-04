import torchvision.models as models
import torch
from torch import nn
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


def get_deep_model(dir_model, num_class):
    device = torch.device('cpu')

    model = models.resnet18()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_class)
    checkpoint_shape = torch.load(dir_model, map_location=device)
    model.load_state_dict(checkpoint_shape['state_dict'])

    return model


def image_process(dir_dataset_color, dir_dataset_shape, dir_dataset_texture):
    mytransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), ])

    color_img1 = Image.open(dir_dataset_color)
    color_img2 = color_img1.convert('RGB')
    color_img3 = mytransform(color_img2)

    shape_img1 = Image.open(dir_dataset_shape)
    shape_img2 = shape_img1.convert('RGB')
    shape_img3 = mytransform(shape_img2)

    texture_img1 = Image.open(dir_dataset_texture)
    texture_img2 = texture_img1.convert('RGB')
    texture_img3 = mytransform(texture_img2)

    return color_img3, shape_img3, texture_img3


def get_deep_results(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)
    soft_output = F.softmax(outputs, dim=1)

    return soft_output


def choquet_integral(output_deepnets, fuzzy_measures, class_list):
    num_integral = len(output_deepnets)
    FI_results = torch.zeros([1, num_class], dtype=torch.float)
    g_matrix = torch.zeros([1, num_integral], dtype=torch.float)

    for i_class in range(num_class):
        # Compute g1, g2, g3, g12, g23, g13, g123
        class_name = class_list[i_class]
        g_group = fuzzy_measures[class_name]

        integral_matrix = torch.cat(
            (output_deepnets['output_shape'][:, i_class].unsqueeze(1),
             output_deepnets['output_texture'][:, i_class].unsqueeze(1),
             output_deepnets['output_color'][:, i_class].unsqueeze(1)), dim=1)
        sorted_output, sorted_indices = torch.sort(integral_matrix, descending=True, dim=-1)

        if sorted_indices[0, :].equal(torch.tensor([0, 1, 2])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g1'], g_group['g12'] - g_group['g1'], g_group['g123'] - g_group['g12']],
                dtype=torch.float)
        elif sorted_indices[0, :].equal(torch.tensor([0, 2, 1])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g1'], g_group['g13'] - g_group['g1'], g_group['g123'] - g_group['g13']],
                dtype=torch.float)
        elif sorted_indices[0, :].equal(torch.tensor([1, 0, 2])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g2'], g_group['g12'] - g_group['g2'], g_group['g123'] - g_group['g12']],
                dtype=torch.float)
        elif sorted_indices[0, :].equal(torch.tensor([1, 2, 0])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g2'], g_group['g23'] - g_group['g2'], g_group['g123'] - g_group['g23']],
                dtype=torch.float)
        elif sorted_indices[0, :].equal(torch.tensor([2, 0, 1])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g3'], g_group['g13'] - g_group['g3'], g_group['g123'] - g_group['g13']],
                dtype=torch.float)
        elif sorted_indices[0, :].equal(torch.tensor([2, 1, 0])):
            g_matrix[0, :] = torch.tensor(
                [g_group['g3'], g_group['g23'] - g_group['g3'], g_group['g123'] - g_group['g23']],
                dtype=torch.float)
        else:
            print('A mistake in Choquet integral, but you are so great.')

        fuzzy_integral = sorted_output * g_matrix
        print(i_class)
        print(class_name)
        print(sorted_indices)
        print('(0 is shape, 1 is texture, 2 is color.)')
        print(g_matrix)
        FI_results_one_class = torch.sum(fuzzy_integral, dim=1)
        FI_results[0, i_class] = FI_results_one_class

    return FI_results


def main():

    model_color = get_deep_model(dir_model_color, num_class)
    model_shape = get_deep_model(dir_model_shape, num_class)
    model_texture = get_deep_model(dir_model_texture, num_class)
    model_color.eval()
    model_shape.eval()
    model_texture.eval()

    color_img, shape_img, texture_img = image_process(dir_dataset_color, dir_dataset_shape, dir_dataset_texture)
    color_img = color_img.unsqueeze(0)
    shape_img = shape_img.unsqueeze(0)
    texture_img = texture_img.unsqueeze(0)
    output_shape = get_deep_results(shape_img, model_shape)
    output_texture = get_deep_results(texture_img, model_texture)
    output_color = get_deep_results(color_img, model_color)

    output_deepnets = {'output_color': output_color, 'output_shape': output_shape, 'output_texture': output_texture}

    # The value of fuzzy measures on fold 2 color-biased dataset, ChI-DE
    # g1 is shape, g2 is texture, g3 is color
    fuzzy_measures = {
        'n01531178': {'g1': 0.35450942649974426, 'g2': 0.41385761189725656, 'g3': 0.8739046939469108,
                      'g12': 0.633878002646221, 'g13': 0.944426110444368, 'g23': 0.956232053496744, 'g123': 1.0},
        'n01537544': {'g1': 0.28105511459686194, 'g2': 0.370296691984966, 'g3': 0.492125535056457,
                      'g12': 0.614622227988636, 'g13': 0.724366917560063, 'g23': 0.798108992452186, 'g123': 1.0},
        'n01558993': {'g1': 0.2944374843171366, 'g2': 0.6821016038933396, 'g3': 0.5903033620935982,
                      'g12': 0.808679956701244, 'g13': 0.739472444866989, 'g23': 0.935872374296662, 'g123': 1.0},
        'n01560419': {'g1': 0.6592549002093172, 'g2': 0.3988813500613485, 'g3': 0.4873106998117892,
                      'g12': 0.845663179475247, 'g13': 0.886988658646057, 'g23': 0.729135347241727, 'g123': 1.0},
        'n01580077': {'g1': 0.21935926824731394, 'g2': 0.8196105586337116, 'g3': 0.3649412828676982,
                      'g12': 0.895636747404698, 'g13': 0.520479804358569, 'g23': 0.946092995794036, 'g123': 1.0},
        'n01582220': {'g1': 0.006478063389602107, 'g2': 0.6180170283291642, 'g3': 0.45677771741611695,
                      'g12': 0.623368212052086, 'g13': 0.462422901615738, 'g23': 0.995336813491144, 'g123': 1.0},
        'n01818515': {'g1': 0.4725019679489889, 'g2': 0.5185813204872937, 'g3': 0.5902362775413497,
                      'g12': 0.790999452198954, 'g13': 0.835007831703081, 'g23': 0.858878451299706, 'g123': 1.0},
        'n01828970': {'g1': 0.6474974178780873, 'g2': 0.18222990428959704, 'g3': 0.6398564144126575,
                      'g12': 0.735604388895908, 'g13': 0.956863823902377, 'g23': 0.729074113658033, 'g123': 1.0},
        'n01843065': {'g1': 0.6092706391593772, 'g2': 0.4364153328362711, 'g3': 0.48554095740600844,
                      'g12': 0.835329356431479, 'g13': 0.860775931837646, 'g23': 0.754318551028989, 'g123': 1.0},
        'n01860187': {'g1': 0.3417429249642382, 'g2': 0.4424721555878524, 'g3': 0.45623034707282084,
                      'g12': 0.707509960765331, 'g13': 0.718883089794784, 'g23': 0.796300372678794, 'g123': 1.0},
        'n02002556': {'g1': 0.4656590078514493, 'g2': 0.5237916122942745, 'g3': 0.4092757803460663,
                      'g12': 0.821852356094843, 'g13': 0.743978299597746, 'g23': 0.785762370837879, 'g123': 1.0},
        'n02009912': {'g1': 0.4788655908984066, 'g2': 0.23409325251438307, 'g3': 0.6499163279516358,
                      'g12': 0.635494207566562, 'g13': 0.913715781169728, 'g23': 0.778874585109938, 'g123': 1.0},
        'n07742313': {'g1': 0.23588093743967953, 'g2': 0.6035893052296422, 'g3': 0.42489556602892564,
                      'g12': 0.758463679519096, 'g13': 0.603752084375099, 'g23': 0.882566638630825, 'g123': 1.0},
        'n07745940': {'g1': 0.053048195377812336, 'g2': 0.8043743656803934, 'g3': 0.9096276177871652,
                      'g12': 0.815709736760483, 'g13': 0.915504820880785, 'g23': 0.998744155621039, 'g123': 1.0},
        'n07747607': {'g1': 0.109542949340778, 'g2': 0.29330790035657395, 'g3': 0.6870691958489097,
                      'g12': 0.393295350140762, 'g13': 0.774228536059830, 'g23': 0.920443623208908, 'g123': 1.0},
        'n12620546': {'g1': 0.7198923198556921, 'g2': 0.40593394131617067, 'g3': 0.5368427747655149,
                      'g12': 0.870953211549266, 'g13': 0.919668539583443, 'g23': 0.752711130717792, 'g123': 1.0},
        'n12768682': {'g1': 0.5701331470247739, 'g2': 0.5920721859557425, 'g3': 0.9886512575155701,
                      'g12': 0.825464649688982, 'g13': 0.996489623785834, 'g23': 0.996791248456591, 'g123': 1.0}}

    class_list = ['n01531178', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01818515',
                  'n01828970', 'n01843065', 'n01860187', 'n02002556', 'n02009912', 'n07742313', 'n07745940',
                  'n07747607', 'n12620546', 'n12768682']
    choquet_integral_results = choquet_integral(output_deepnets, fuzzy_measures, class_list)
    print('The Choquet integral results are:')
    print(choquet_integral_results)
    value, index = torch.max(choquet_integral_results, dim=1)
    print('The final results is: ')
    print(class_list[index])
    print('With the corresponding prediction:')
    print(value)



if __name__ == '__main__':
    # Parameters
    dataset_name = 'color_biased_dataset'
    image_index = 'n01828970_11178'
    dir_root = os.path.join('data', dataset_name, image_index)
    num_class = 17

    dir_model_color = os.path.join('data', dataset_name, 'model/color_resnet18', '22.pth')
    dir_model_shape = os.path.join('data', dataset_name, 'model/shape_resnet18', '17.pth')
    dir_model_texture = os.path.join('data', dataset_name, 'model/texture_resnet18', '29.pth')

    color_address = image_index + '.jpg'
    shape_address = image_index + '.png'
    texture_address = image_index + '.JPEG'
    dir_dataset_color = os.path.join(dir_root, 'color', color_address)
    dir_dataset_shape = os.path.join(dir_root, 'shape', shape_address)
    dir_dataset_texture = os.path.join(dir_root, 'texture', texture_address)

    main()
