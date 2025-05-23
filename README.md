# Explainable-fuzzy-integral
The code for the paper "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision".

**Reference:**  
Y. Wang, W. Pedrycz, H. Ishibuchi and J. Zhu, "Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision," *IEEE Transactions on Fuzzy Systems*, vol. 33, no. 1, pp. 156-167, 2025.

@ARTICLE{10613462,   
  author={Wang, Yifan and Pedrycz, Witold and Ishibuchi, Hisao and Zhu, Jihua},   
  journal={IEEE Transactions on Fuzzy Systems},    
  title={Fusion of Explainable Deep Learning Features Using Fuzzy Integral in Computer Vision},    
  year={2025},   
  volume={33},   
  number={1},   
  pages={156-167}   
}

The paper can be downloaded from https://ieeexplore.ieee.org/document/10613462.

**Introduction:**   
In this study, the effectiveness of eight fuzzy integral fusion algorithms for enhancing classification accuracy in computer vision has been explored. Computational experiments show that fuzzy integral fusion can improve classification accuracy by 14.6% compared with an individual deep neural network on subsets derived from the ImageNet dataset. In our experiments, original images are transformed into shape, texture, and color images. Then, ResNet18 is employed to extract shape, texture, and color features from these images respectively. Through fuzzy integral fusion, the contributions of shape, texture, and color features for different classes could be clearly evaluated. As a result, the proposed approaches not only achieve higher classification accuracy but also provide explainability. Specifically, given an image pattern, the proposed approaches have the ability to provide a convincing explanation of how this input pattern is classified into a specific class.

## Part 1: Fuzzy Integral
Before using the fuzzy integral to combine shape, texture, and color features, we must train deep learning networks for each individual feature.
### Training Deep learning networks
I used PyTorch 3.8 to run the code, but I think other versions, such as PyTorch 3.10, should also work.   
Please refer to the folder (i.e., 1_fuzzy_integral_fold1/Training_DNNs).   

```
python main_shape.py 
```

We can obtain a ResNet18 model for the shape images. 

The ResNet18 model with the best validation accuracy is stored in a pth file named ‘16.pth’. You can find this file located at ‘1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/’.

The training process and results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/shape_resnet18/”.

```
python main_texture.py 
```

We can obtain a ResNet18 model for the texture images. 

The ResNet18 model with the best validation accuracy is stored in a pth file named ‘23.pth’. You can find this file located at ‘1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/’.

The training process and results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/texture_resnet18/”.


```
python main_color.py 
```

We can obtain a ResNet18 model for the color images. 

The ResNet18 model with the best validation accuracy is stored in a pth file named ‘8.pth’. You can find this file located at ‘1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/’.

The training process and results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/color_resnet18/”.

```
python main_ori.py 
```

We can obtain a ResNet18 model for the original images. 

The ResNet18 model with the best validation accuracy is stored in a pth file named ‘25.pth’. You can find this file located at ‘1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/’.

The training process and results are shown in a text file named “log.txt” located at “1_fuzzy_integral_fold1/data/all_datasets/model/original_resnet18/”.

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

The required package can be installed with the following command:  
```
pip install sympy
```

Then:   
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

The required package can be installed with the following command:  
```
pip install cvxopt
```

Then:   
```
python main_ChI_QP_all.py
```

We can obtain the fuzzy fusion results of ChI-QP.

The training process and results are shown in a text file named "ChI-QP.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### iCHIMP

```
python main_iCHIMP_all.py
```

We can obtain the fuzzy fusion results of iCHIMP.

The training process and results are shown in a text file named "iCHIMP.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### ChI-DE

The required package can be installed with the following command:  
```
pip install geatpy
```

Then:   
```
python main_ChI_DE_all.py
```

We can obtain the fuzzy fusion results of ChI-DE.

The training process and results are shown in a text file named "ChI-DE.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

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

The training process and results are shown in a text file named "FI_CNN_F1_score.txt" located at "1_fuzzy_integral_fold1/results/all_datasets/".

#### Conclusion
Here, all the fuzzy fusion results are evaluated on the combined dataset (fold 1).

The combined dataset refers to the combination of three datasets: shape, texture, and color-biased datasets.   


| Method | Accuracy |
|:-------|:--------|
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

#### Please note:   
* I learned the foundational theory of the fuzzy integral through this video: https://www.youtube.com/watch?v=467iAANUTvw. The implementation of four algorithms (i.e., Shared SI, SI for each class, Shared ChI, and ChI for each class) is based on this video. If you’re not a researcher in the fuzzy community, I highly recommend checking out this video. I also have learned so much from https://github.com/scottgs/FuzzyFusion_DeepLearning_Tutorial and https://github.com/scottgs/fi_library.      
* ChI-QP is a copy from https://github.com/B-Mur/choquet-integral.   
* iCHIMP is a copy from https://github.com/aminb99/choquet-integral-NN.  
* In ChI-DE, the implementation of the evolutionary algorithm is based on Geatpy: https://github.com/geatpy-dev/geatpy.   
* The implementation of FI-CNN is based on a TCSVT paper. The original code can be downloaded from https://github.com/theavicaster/fuzzy-integral-cnn-fusion-3d-har.   

## Part 2: Explainability
Please refer to the folder (i.e., 2_explainability).

Here, we show how an input image can be classified into a specific class using fuzzy integral fusion.

Specially, this provides a detailed classification process for a given image (i.e., n01828970_11178) from the color-biased dataset (fold 2), as executed by ChI-DE.

```
python main_ChI_DE_color_fold2.py
```

The classification process (including contributions and predictions) is shown in a text file named "explainability_results.txt" located at "2_explainability/".

Prediction results of this given image from the color-biased dataset (Fold 2) are shown as follows:

|No.|Class name|Shape|Texture|Color|ChI-DE|
|:-------|:---------|:--------|:---------|:-----------|:------------|
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

Contributions of different features as determined by ChI-DE are shown as follows:

|No.|Class name|Shape|Texture|Color|
|:-------|:---------|:--------|:---------|:-----------|
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

## Part 3: Image Preprocessing

The three feature-biased datasets can be downloaded from http://ilab.usc.edu/andy/dataset/hve.zip.

The code is a copy from https://github.com/gyhandy/Humanoid-Vision-Engine and http://martin-hebart.de/code/imscramble.m.

Here, we provide simple examples demonstrating how to use this code.

### Select A Mask 

We can segment an image using any segmentation algorithm.

Please refer to the folder (i.e., 3_image_preprocessing/1_select_mask).

```
python show_mask_of_a_fig.py
```

The original image is '3_image_preprocessing/1_select_mask/original_images.jpg'.

We can obtain 11 masks of this image. The masks are saved in "3_image_preprocessing/1_select_mask/mask".

```
python select_mask.py
```

The selected mask is the file "3_image_preprocessing/1_select_mask/preprocessed_images/Segement_GradCam/mask/boat/example.jpg".

The foreground image is the file "3_image_preprocessing/1_select_mask/preprocessed_images/Segement_GradCam/img/boat/example.jpg".

The images computed by GradCAM are saved to the folder "3_image_preprocessing/1_select_mask/cam_images".

### Shape Image Generation

Please refer to the folder (i.e., 3_image_preprocessing/2_shape_image_generation).

Download the model from "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt" and save it in the folder "3_image_preprocessing/2_shape_image_generation/DPT/weights".

```
python run_monodepth.py
```

The image with depth information can be found in "3_image_preprocessing/2_shape_image_generation/ori_img_monodepth/boat/example.png".

```
python generate_shape_feature.py
```

The generated shape image is shown in "3_image_preprocessing/2_shape_image_generation/generated_shape_images/boat/example.png".

### Texture Image Generation

Please refer to the folder (i.e., 3_image_preprocessing/3_texture_image_generation).

```
python generate_texture_feature.py
```

The generated texture image is shown in "3_image_preprocessing/3_texture_image_generation/texture_img.jpg".

### Color Image Generation

Please refer to the folder (i.e., 3_image_preprocessing/4_color_image_generation)

Run main.m

The generated color image is shown in "3_image_preprocessing/4_color_image_generation/result.jpg".

## Part 4: Future Research Directions

The proposed approaches show significant potential for further application in the field of computer vision. The limitations and future research directions are divided into four parts:

1) Our study primarily deals with simple images, each containing a single object. However, real-world images often include multiple objects. For instance, an image could simultaneously highlight a person, a dog, and a cat as significant foreground objects. In certain scenarios, background objects also contribute to the recognition of foreground objects. For example, planes and birds are typically associated with the sky, while horses, sheep, and flowers are associated with grass. Such background objects (i.e., sky and grass) have been ignored in our study. In the future, we consider incorporating causal relationships as beneficial features before making final decisions. Utilizing a relational graph model to establish connections among all objects in an image is a good choice.
2) When compared to individual DNNs, the fuzzy integral strategy can significantly enhance the classification performance. However, a large gap still exists in terms of classification accuracy. For instance, ResNet18 achieves 95.21% on the combined dataset (fold 1) on foreground images, whereas achieves 89.93% based on fuzzy integral fusion. Enhancing the proposed explainable classifier is an interesting research direction. On the one hand, we can use some tricks to improve classification accuracy such as data augmentation, integrating different loss functions, and dropout during the training process. Enhancing the accuracy of a single classifier will subsequently improve the results of fused classification. On the other hand, in this study, the DNNs are frozen in the fusion process. Future studies could investigate the fine-tuning of parameters in both DNNs and fuzzy integral fusion with more GPUs and CPUs.
3) In our experiments, the inputs for the fuzzy integral are real numbers. It would be interesting to extend the proposed approaches to general fuzzy set-valued integral fusion. Additional fuzzy fusion techniques could also be explored for the fusion of shape, texture, and color features in computer vision such as Takagi–Sugeno–Kang fuzzy systems. In this way, the fusion process extends beyond the decision level and can be applied to feature-level fusion.
4) Feature extraction remains a challenging topic. When it comes to shape features, we often capture the edges of foreground objects. However, relying solely on edges falls short when describing complex shapes like “a cat with two pointed ears and a round face”. For texture features, the small size of patches often obscures details, making it challenging to describe fine patterns such as stripes and spots on objects. Beyond shape, texture, and color features, how to extract some visual attributes (i.e., domestic, small, and stripe) that relate to the object and how the object is described by these attributes is an interesting research direction. Existing studies, such as the hierarchical criteria network, offer a promising approach for learning useful attributes and descriptions. For instance, they can help describe a tabby cat as a domestic feline with stripes, dots, or lines.

(The TFS paper also explores future research directions.)

Thanks for your attention~








