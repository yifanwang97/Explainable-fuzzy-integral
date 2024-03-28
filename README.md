# Explainable_fuzzy_integral
The code for the paper "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision".

## Part 1: Fuzzy Integral
Before using fuzzy integral combine the shape, texture, and color features, we need to training deep learning networks for each single feature.
### Training Deep learning networks
Please refer to the folder (i.e., 1_fuzzy_integral_fold1/Training_DNNs).

#### Run "main_shape.py". 

We can get a ResNet18 model for the shape images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/log.txt".

#### Run "main_texture.py". 

We can get a ResNet18 model for the texture images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/log.txt".

#### Run "main_color.py". 

We can get a ResNet18 model for the color images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/log.txt".

#### Run "main_ori.py". 

We can get a ResNet18 model for the original images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/log.txt".

### Fuzzy Integral Fusion
Please refer to the folder (i.e., 1_fuzzy_integral_fold1).

#### Baseline
Run "testing_acc_test.py". 

We can get the classification accuracy of trained ResNet18 models on the single-feature testing datasets. 

The testing results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/testing_accuracy.txt".

You can also run "testing_acc_train.py" and "testing_acc_val.py".

The training results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/training_accuracy.txt".

The validation results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/validation_accuracy.txt".

#### Shared SI & Shared ChI
Run "main_simple_integral.py".

We can obtain the fuzzy fusion results of two methods: Shared SI and Shared ChI.

SI means Sugeno Integral; ChI means Choquet Integral.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/Shared_SI_ChI.txt".

#### SI for each class & ChI for each class
(1) Run "computing_contribution_val.py".
This function computes the fuzzy density values for each class. 

The results are shown in a json file "".






