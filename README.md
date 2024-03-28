# Explainable_fuzzy_integral
The code for the paper "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision".

## Part 1: Fuzzy Integral
Before using fuzzy integral combine the shape, texture, and color features, we need to training deep learning networks for each single feature.
### Training Deep learning networks
Please refer to the folder (i.e., 1_fuzzy_integral_fold1/Training_DNNs).

```
python main_shape.py 
```

We can get a ResNet18 model for the shape images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/log.txt".

```
python main_texture.py 
```

We can get a ResNet18 model for the texture images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/log.txt".

```
python main_color.py 
```

We can get a ResNet18 model for the color images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/log.txt".

```
python main_ori.py 
```

We can get a ResNet18 model for the original images. 

The training results are shown in a txt file "1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/log.txt".

### Fuzzy Integral Fusion
Please refer to the folder (i.e., 1_fuzzy_integral_fold1).

#### Baseline

```
python testing_acc_test.py 
```

We can get the classification accuracy of trained ResNet18 models on the single-feature testing datasets. 

The testing results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/testing_accuracy.txt".

```
python testing_acc_train.py
```
The training results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/training_accuracy.txt".

```
python testing_acc_val.py
```

The validation results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/validation_accuracy.txt".

#### Shared SI & Shared ChI

```
python main_simple_integral.py
```

We can obtain the fuzzy fusion results of two methods: Shared SI and Shared ChI.

SI means Sugeno Integral; ChI means Choquet Integral.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/Shared_SI_ChI.txt".

#### SI for each class & ChI for each class

```
python computing_contribution_val.py
```

This function computes the fuzzy density values for each class. 

The results are shown in a json file "1_fuzzy_integral_fold1/data/all_datasets/contribution/contribution_each_class_val.json".

```
python main_gi_for_each_class.py
```

We can obtain the fuzzy fusion results of two methods: SI for each class and ChI for each class.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/gi_for_each_class_SI_ChI.txt".

#### ChI-QP

```
python main_ChI_QP_all.py
```

We can obtain the fuzzy fusion results of ChI-QP.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/ChI-QP.txt".

#### iCHIMP

```
python main_iCHIMP_all.py
```

We can obtain the fuzzy fusion results of iCHIMP.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/iCHIMP.txt".

#### ChI-DE

```
python main_ChI_DE_all.py
```

We can obtain the fuzzy fusion results of ChI-DE.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/ChI-DE.txt".

#### FI-CNN

```
python testing_acc_val_F1_all.py
```

In this way, we can obtain the accuracy and F1 score on the validation datasets. 

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/F1_score.txt".

```
python main_FI_CNN_all.py
```

We can obtain the fuzzy fusion results of FI-CNN.

The results are shown in a txt file "1_fuzzy_integral_fold1/results/all_datasets/FI_CNN_F1_score.txt".

#### Conclusion
Here, all results are fuzzy fusion results on the combined dataset (fold 1).

| Method | Accuracy |
|:-------|-----------|
|ResNet18 (shape)|69.97%|
|ResNet18 (texture)|75.33%|
|ResNet18 (color)|66.72%|
|Shared SI|83.74%|
|SI for each class|84.76%|
|Shared ChI|87.81%|
|ChI for each class|87.55%|
|ChI-QP|89.41%|
|iCHIMP|89.65%|
|ChI-DE|87.96%|
|FI-CNN|89.93%|

## Part 2: Explainability
Please refer to the folder (i.e., ).

Here, we show how an input image can be classified into a specific class using fuzzy integral fusion.

Specially, this provides a detailed classification process for a given image from the color-biased dataset (fold 2), as executed by ChI-DE.





