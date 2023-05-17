This code is an implementation of our work "GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples."
[Reference]: Tian Gao and Cheng-Zhong Xu and Le Zhang and Hui Kong, GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples, arXiv,2023


----------------------------------------------------------------------------------------------------------------------------------------------------------
# [GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples](https://arxiv.org/abs/2305.07931)
### Introduction
Vision Transformer (ViT) has performed remarkably in various computer vision tasks. Nonetheless, affected by the massive amount of parameters, ViT usually suffers from serious overfitting problems with a relatively limited number of training samples. In addition, ViT generally demands heavy computing resources, which limit its deployment on resource-constrained devices. As a type of model-compression method,  model binarization is potentially a good choice to solve the above problems. Compared with the full-precision one, the model with the binarization method replaces complex tensor multiplication with simple bit-wise binary operations and represents full-precision model parameters and activations with only 1-bit ones, which potentially solves the problem of model size and computational complexity, respectively. In this paper, we investigate a binarized ViT model. Empirically, we observe that the existing binarization technology designed for Convolutional Neural Networks (CNN) cannot migrate well to a ViT's binarization task. We also find that the decline of the accuracy of the binary ViT model is mainly due to the information loss of the Attention module and the Value vector. Therefore, we propose a novel model binarization technique, called Group Superposition Binarization (GSB), to deal with these issues. Furthermore, in order to further improve the performance of the binarization model, we have investigated the gradient calculation procedure in the binarization process and derived more proper gradient calculation equations for GSB to reduce the influence of gradient mismatch. Then, the knowledge distillation technique is introduced to alleviate the performance degradation caused by model binarization. Analytically, model binarization can limit the parameter’s search space during parameter updates while training a model. Therefore, the binarization process can actually play an implicit regularization role and help solve the problem of overfitting in the case of insufficient training data. Experiments on three datasets with limited numbers of training samples demonstrate that the proposed GSB model achieves state-of-the-art performance among the binary quantization schemes and exceeds its full-precision counterpart on some indicators.
### Datasets
* CIFAR-100 Dataset [Download link](http://www.cs.toronto.edu/~kriz/cifar.html)
* Oxford-Flowers102 Datasets （Small dataset, only 20 samples for one class）[Homepage link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102)
* Chaoyang Datasets (Medical image w/ label noise) [Homepage link](https://bupt-ai-cz.github.io/HSA-NRL/)  
### Environment and Dependencies

### Training and Evaluation
### Tips
   Any problem, please contact the first author (Email: gaotian970228@njust.edu.cn).
### Citation
If you find this work useful, please consider citing:

    @ARTICLE{gao2023gsb,
          title={GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples}, 
          author={Tian Gao and Cheng-Zhong Xu and Le Zhang and Hui Kong},
          year={2023},
          eprint={2305.07931},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
          }
### License
Our code is released under the MIT License (see LICENSE file for details).
### Acknowledgement
Our code refers to Q-ViT(https://github.com/YanjingLi0202/Q-ViT), BiT(https://github.com/facebookresearch/bit) and DeiT(https://github.com/facebookresearch/deit).




### Reference
* Zhu, Chuang and Chen, Wenkai and Peng, Ting and Wang, Ying and Jin, Mulan, Hard Sample Aware Noise Robust Learning for Histopathology Image Classification, IEEE transactions on medical imaging, 2021
* Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
* Nilsback, M-E. and Zisserman, A. Automated flower classification over a large number of classes, Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)
