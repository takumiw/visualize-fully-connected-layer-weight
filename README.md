# visualize-fully-connected-layer-weight
> Keras implementation to visualize outputs and weights of fully connected layer.

This repository contains keras (tensorflow.keras) implementation to visualize outputs and weights of fully connected layer of common CNN (VGG8) and ArcFace [1] using *Fashion MNIST* dataset [1].

## dataset
Fashion MNIST dataset includes 10 classes in 4 categories:
- Tops: T-shirt/top, Pullover, Dress, Coat, Shirt
- Bottoms: Trouser
- Shoes: Sandal, Sneaker, Ankle boot
- Bags: Bag

## CNN (VGG8)
Figure 1 show the output vectors of the fully connected layer of VGG8 trained on Fashion MNIST dataset. Figure 2 show the weight vectors of the fully connected layer of VGG8 trained on Fashion MNIST dataset.

<img src="https://user-images.githubusercontent.com/30923675/97911554-c9045d00-1d8e-11eb-93b8-7ef90a8c6dd4.png"> <img src="https://user-images.githubusercontent.com/30923675/97911681-07018100-1d8f-11eb-9e71-65a8cacb31b5.png">

Figure 1. Output of fully connected layer in (a) train and (b) test data

<img src="https://user-images.githubusercontent.com/30923675/97913629-ee469a80-1d91-11eb-80b7-4072a8c58aec.png">

Figure 2. Weight of fully connected layer


## ArcFace (employs VGG8 as embedding network)
Figure 3 show the output vectors of the fully connected layer of VGG8 trained on Fashion MNIST dataset. Figure 4 show the weight vectors of the fully connected layer of VGG8 trained on Fashion MNIST dataset.

<img src="https://user-images.githubusercontent.com/30923675/97913533-c3f4dd00-1d91-11eb-8281-cd8aa7d39ce6.png"> <img src="https://user-images.githubusercontent.com/30923675/97913662-fa325c80-1d91-11eb-9388-045b5a66f920.png">

Figure 3. Output of fully connected layer in (a) train and (b) test data

<img src="https://user-images.githubusercontent.com/30923675/97913551-cbb48180-1d91-11eb-9a82-2ba144da2d6e.png">

Figure 4. Weight of fully connected layer

# Reference
[1] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. CVPR.  
[2] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.  
[3] https://github.com/4uiiurz1/keras-arcface