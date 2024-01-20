This code is an implementation of our work "GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples."

[Reference]: Tian Gao and Cheng-Zhong Xu and Le Zhang and Hui Kong, GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples, Neural Networks, 2024


----------------------------------------------------------------------------------------------------------------------------------------------------------
# [GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples](https://arxiv.org/abs/2305.07931)
### Introduction
Vision Transformer (ViT) has performed remarkably in various computer vision tasks. Nonetheless, affected by the massive amount of parameters, ViT usually suffers from serious overfitting problems with a relatively limited number of training samples. In addition, ViT generally demands heavy computing resources, which limit its deployment on resource-constrained devices. As a type of model-compression method,  model binarization is potentially a good choice to solve the above problems. Compared with the full-precision one, the model with the binarization method replaces complex tensor multiplication with simple bit-wise binary operations and represents full-precision model parameters and activations with only 1-bit ones, which potentially solves the problem of model size and computational complexity, respectively. In this paper, we investigate a binarized ViT model. Empirically, we observe that the existing binarization technology designed for Convolutional Neural Networks (CNN) cannot migrate well to a ViT's binarization task. We also find that the decline of the accuracy of the binary ViT model is mainly due to the information loss of the Attention module and the Value vector. Therefore, we propose a novel model binarization technique, called Group Superposition Binarization (GSB), to deal with these issues. Furthermore, in order to further improve the performance of the binarization model, we have investigated the gradient calculation procedure in the binarization process and derived more proper gradient calculation equations for GSB to reduce the influence of gradient mismatch. Then, the knowledge distillation technique is introduced to alleviate the performance degradation caused by model binarization. Analytically, model binarization can limit the parameter’s search space during parameter updates while training a model. Therefore, the binarization process can actually play an implicit regularization role and help solve the problem of overfitting in the case of insufficient training data. 
### Datasets
* CIFAR-100 Dataset [Download link](http://www.cs.toronto.edu/~kriz/cifar.html)
* Oxford-Flowers102 Datasets （Small dataset, only 20 samples for one class）[Homepage link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102)
* Chaoyang Datasets (Medical image w/ label noise) [Homepage link](https://bupt-ai-cz.github.io/HSA-NRL/)  
### Environment and Dependencies
Our code was tested with Python 3.8.12, Pytorch 1.10.0,and cuda 11.3  

Required python packages：
* PyTorch (version 1.10.0)
* numpy
* timm
### Training and Evaluation
Note that our training code refers to Q-ViT. For more details of the training code please refer to [here](https://github.com/YanjingLi0202/Q-ViT).

* Train the network
     
     ```
     # To train model with one GPU
     # stage 1
     python train_stage1.py --batch-size (Choose the appropriate batchsize according to the GPU)  --teacher-path  （Path to the folder where the parameter file of teacher network  is stored） --teacher_p （Path to the parameter file  of teacher network） --data-path （path to the dataset） --data-set （The name of the data set, please refer to file datasets.py for details）--output_dir (Path to store output)
     # stage 2
     python train_stage2.py --batch-size ** --teacher-path ** --teacher_p ** --data-path ** --data-set ** --output_dir** --finetune (Fill in the path of the file output by stage1 here)
     
     # To train model with Multiple GPUs
     # stage 1
     python -m torch.distributed.launch --nproc_per_node=2 （Change to the total number of your GPU） --use_env train_stage1.py  --batch-size ** --teacher-path ** --teacher_p ** --data-path ** --data-set ** --output_dir**
     # stage 2
     python -m torch.distributed.launch --nproc_per_node=2 （Change to the total number of your GPU） --use_env train_stage2.py  --batch-size ** --teacher-path ** --teacher_p ** --data-path ** --data-set ** --output_dir** --finetune **   
     # ** means your parameters
     ```
 * Test the network
     ```
     python test.py -data-path ** --data-set **  --resume （path to the file of model parameter ）
     ```
### Tips
   Any problem, please contact the first author (Email: gaotian970228@njust.edu.cn).
   
   Our pre-trained model can be downloaded at the following link
   * CIFAR  
   
      [teacher model](https://drive.google.com/file/d/1h_NWxG0-TcUU6-CbUvzclljfzxBlzCn4/view?usp=share_link)
      
      [stage1](https://drive.google.com/file/d/1Pa5tprJuP9dFmq-F-hS-o_VnUb9foKSq/view?usp=share_link)
      
      [stage2](https://drive.google.com/file/d/15gHr5m_9ukbcGssx1CBymrnL6gLNJ0-v/view?usp=share_link)
   * Oxford-Flowers102
      
         
      [teacher model](https://drive.google.com/file/d/13N_uqWNCDwj5g6AgBXqqxzVBs1pBPnGp/view?usp=share_link)
      
      [stage1](https://drive.google.com/file/d/1fPccxP5BEdiexw3EqwsZuCvgNRjMcLUL/view?usp=share_link)
      
      [stage2](https://drive.google.com/file/d/1c5NLlvShWbB9RIlP70F0zKEA1KbCcXMU/view?usp=share_link)
   * Chaoyang Datasets
   
         
      [teacher model](https://drive.google.com/file/d/1jOBObIn3fzRQvRrT_RSbWVJNKeN_iAro/view?usp=share_link)
      
      [stage1](https://drive.google.com/file/d/1_tGscV2lRYv1E9ueDdn4NEonN9bM8OpQ/view?usp=share_link)
      
      [stage2](https://drive.google.com/file/d/1xSLqDuAS6V_B9YYbk4Jrgjyq_YHNmEnt/view?usp=share_link)
### Citation
If you find this work useful, please consider citing:

    @ARTICLE{gao2023gsb,
          title={GSB: Group Superposition Binarization for Vision Transformer with Limited Training Samples}, 
          author={Tian Gao and Cheng-Zhong Xu and Le Zhang and Hui Kong},
          year={2024},
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
* Li Y, Xu S, Zhang B, et al. Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer. NIPS, 2022.
* Liu Z, Oguz B, Pappu A, et al. Bit: Robustly binarized multi-distilled transformer. NIPS, 2022.
