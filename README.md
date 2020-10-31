# model_pruning
### Model Pruning 

Pruning in deep learning basically used so that we can develop a neural network model that is smaller and more efficient. The goal of this technique is to optimize the model by eliminating the values of the weight tensors. The aim is to get a computationally cost-efficient model that takes a less amount of time in training. The necessity of pruning on one hand is that it saves time and resources while on the other hand is essential for the execution of the model in low-end devices such as mobile and other edge devices. Deep Learning Algorithms like Convolutional Neural Networks (CNNs) suffer from different issues, such as computational complexity and the number of parameters. Hence the goal here is to test different methodologies for pruning a deep learning model and suggest a different approach for Pruning.
![alt text](https://github.com/guramritpalsaggu/model_pruning/blob/main/pruning.png)

Two Types of Techniques has been experiemnts here:

- Method1:
In this technique we aim to remove the weights of neurons which are not necessary while making inference from the model, therefore making the model more efficient. We can also view this task as feature selecting i.e selecting only neurons that are necessary for model inference. Lasso regression which uses l1 norm in regression also is known for the feature selection, we will use l1 norm to determine the weights of neurons to be removed.
Inspired by: lasso regression and Insipired by: https://github.com/Raukk/tf-keras-surgeon

- Method2:
Using tensorflow method insipired by the paper "To prune, or not to prune: exploring the efficacy of pruning for model compression"
blog: https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html
