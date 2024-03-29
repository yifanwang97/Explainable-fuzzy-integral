# Explainable_fuzzy_integral
The code for the paper "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision".

The three feature-biased datasets can be downloaded from http://ilab.usc.edu/andy/dataset/hve.zip.

## Part 1: Fuzzy Integral
Before using the fuzzy integral to combine shape, texture, and color features, we must train deep learning networks for each individual feature.
### Training Deep learning networks
Please refer to the folder (i.e., 1_fuzzy_integral_fold1/Training_DNNs).

```
python main_shape.py 
```

We can obtain a ResNet18 model for the shape images. 

The training results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/”.

```
python main_texture.py 
```

We can obtain a ResNet18 model for the texture images. 

The training results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/”.

```
python main_color.py 
```

We can obtain a ResNet18 model for the color images. 

The training results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/”.

```
python main_ori.py 
```

We can obtain a ResNet18 model for the original images. 

The training results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/”.

### Fuzzy Integral Fusion
Please refer to the folder (i.e., 1_fuzzy_integral_fold1).

#### Baseline

```
python testing_acc_test.py 
``` 

We can obtain the classification accuracy of the trained ResNet18 models on the single-feature testing datasets. 

The testing results are shown in a text file named “testing_accuracy.txt” located at “1_fuzzy_integral_fold1/results/all_datasets/”.

```
python testing_acc_train.py
```

The training results are shown in a text file named "training_accuracy.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

```
python testing_acc_val.py
```

The validation results are shown in a text file named "validation_accuracy.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### Shared SI & Shared ChI

```
python main_simple_integral.py
```

We can obtain the fuzzy fusion results of two methods: Shared SI (Sugeno Integral) and Shared ChI (Choquet Integral). 

The results are shown in a text file named "Shared_SI_ChI.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### SI for each class & ChI for each class

```
python computing_contribution_val.py
```

This function computes the fuzzy density values for each class. 

The results are shown in a json file named "contribution_each_class_val.json" located at "1_fuzzy_integral_fold1/data/all_datasets/contribution/".

```
python main_gi_for_each_class.py
```

We can obtain the fuzzy fusion results of two methods: SI for each class and ChI for each class.

The results are shown in a text file named "gi_for_each_class_SI_ChI.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### ChI-QP

```
python main_ChI_QP_all.py
```

We can obtain the fuzzy fusion results of ChI-QP.

The results are shown in a text file named "ChI-QP.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### iCHIMP

```
python main_iCHIMP_all.py
```

We can obtain the fuzzy fusion results of iCHIMP.

The results are shown in a text file named "iCHIMP.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### ChI-DE

```
python main_ChI_DE_all.py
```

We can obtain the fuzzy fusion results of ChI-DE.

The results are shown in a text file named "ChI-DE.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### FI-CNN

```
python testing_acc_val_F1_all.py
```

In this way, we can obtain the accuracy and F1 score on the validation datasets. 

The results are shown in a text file named "F1_score.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

```
python main_FI_CNN_all.py
```

We can obtain the fuzzy fusion results of FI-CNN.

The results are shown in a text file named "FI_CNN_F1_score.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### Conclusion
Here, all the results represent the fuzzy fusion outcomes on the combined dataset (fold 1).

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
Please refer to the folder (i.e., 2_explainability).

Here, we show how an input image can be classified into a specific class using fuzzy integral fusion.

Specially, this provides a detailed classification process for a given image (i.e., n01828970_11178) from the color-biased dataset (fold 2), as executed by ChI-DE.

```
python main_ChI_DE_color_fold2.py
```

The classification process (including contributions and predictions) is shown in a text file named "explainability_results.txt" located at "2_explainability/".

Prediction Results of A Given Image from the Color-biased Dataset (Fold 2)

|No.|Class name|Shape|Texture|Color|ChI-DE|
|:-------|---------|--------|---------|-----------|------------|
|1| Goldfinch | 7.2468e-06 | 2.6133e-03 | 5.4039e-04 | 1.3750e-03 |	
|2| Indigo bunting | 1.1503e-07 | 4.6238e-01 | 1.4650e-03 | 1.7185e-01 |
|3| Robin | 1.1038e-06 | 2.3641e-04 | 1.5929e-03 | 1.0220e-03 |
|4| Bulbul | 7.1790e-06 | 8.4280e-04 | 1.3746e-04 | 3.8352e-04 |
|5| Jay | 2.4887e-06 | 9.7559e-03 | 2.0715e-04 | 8.0223e-03 |
|6| Magpie | 3.5713e-04 | 1.3689e-03 | 1.9123e-05 | 8.5513e-04 |
|7| Macaw | 3.9998e-03 | 7.0415e-05 | 6.9797e-03 | 5.1104e-03 |
|8| Bee eater | 9.4481e-01 | 1.3514e-03 | 3.4885e-01 | 7.1974e-01 |
|9| Jacamar | 3.1552e-06 | 1.8872e-05 | 2.4077e-03 | 1.1749e-03 |
|10| Black swan | 1.4385e-04 | 3.0834e-02 | 1.0924e-03 | 1.4059e-02 |
|11| White stork | 5.9533e-05 | 3.2309e-03 | 7.1100e-04 | 1.8913e-03 |
|12| American egret | 5.0587e-02 | 2.5455e-03 | 1.8586e-04 | 2.4691e-02 |
|13| Granny Smith | 3.5797e-06 | 1.3983e-04 | 1.0046e-02 | 4.3329e-03 |
|14| Strawberry | 7.6811e-06 | 2.8326e-04 | 1.7828e-03 | 1.6470e-03 |
|15| Orange | 8.7682e-06 | 4.8315e-01 | 1.3640e-03 | 1.4257e-01 |
|16| Roschip | 4.1118e-06 | 1.5379e-04 | 6.2114e-01 | 3.3349e-01 |
|17| Buckeye | 8.3045e-08 | 1.0211e-03 | 1.4812e-03 | 1.4727e-03 |

Contributions of Different Features as Determined by ChI-DE

|No.|Class name|Shape|Texture|Color|
|:-------|---------|--------|---------|-----------|
|1| Goldfinch | 4.38% | 41.39% | 54.24% |	
|2| Indigo bunting | 20.19% | 37.03% | 42.78% |	
|3| Robin | 6.41% | 34.56% | 59.03% |
|4| Bulbul | 27.09% | 39.89% | 33.03% |	
|5| Jay | 5.39% | 81.96% | 12.65% |	
|6| Magpie |  0.54% | 61.80% | 37.66% |	
|7| Macaw | 24.48% | 16.50% | 59.02% |
|8| Bee eater | 64.75% | 4.31% | 30.94% |	
|9| Jacamar | 24.57% | 26.88% | 48.55% |
|10| Black swan | 20.37% | 44.25% | 35.38% |
|11| White stork | 21.42% | 52.38% | 26.20% |
|12| American egret | 47.89% | 15.66% | 36.45% |	
|13| Granny Smith | 11.74% | 45.77% | 42.49% |	
|14| Strawberry |  0.13% | 8.91% | 90.96% |	
|15| Orange | 7.96% | 29.33% | 62.71% |	
|16| Roschip | 24.73% | 21.59% | 53.68% |	
|17| Buckeye | 0.32% | 0.81% | 98.87% |

	




