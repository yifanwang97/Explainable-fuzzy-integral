# Explainable_fuzzy_integral
The code for the paper "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision".

## Part 1: Fuzzy Integral
Before using fuzzy integral combine the shape, texture, and color features, we need to training deep learning networks for each single feature.
### Training Deep learning networks
Please refer to the folder (i.e., 1_fuzzy_integral_fold1/Training_DNNs).

Run "main_shape.py". We can get a ResNet18 model for the shape images. The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/log.txt".

Run "main_texture.py". We can get a ResNet18 model for the texture images. The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/log.txt".

Run "main_color.py". We can get a ResNet18 model for the color images. The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/log.txt".

Run "main_ori.py". We can get a ResNet18 model for the original images. The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/log.txt".

