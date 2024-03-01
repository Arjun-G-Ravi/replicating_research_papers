# AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks
#### - Aditya Devarakonda, Maxim Naumov, Michael Garland

## Abstract: 
    Training deep neural networks with Stochastic Gradient Descent, or its variants, requires careful choice of both learning rate and batch size. While smaller batch sizes generally converge in fewer training epochs, larger batch sizes offer more parallelism and hence better computational efficiency. We have developed a new training approach that, rather than statically choosing a single batch size for all epochs, adaptively increases the batch size during the training process. Our method delivers the convergence rate of small batch sizes while achieving performance similar to large batch sizes.

## Summary
- Lower batch size means that the model needs less memory and is faster to converge. This comes at the cost of model being less efficient.
- Higher batch size is slower to converge and requires more memory. The rule of thumb to decide batch size is to choose the highest possible batch size that fits in your memory. 
- A solution to this problem is to `adaptively increase the batch size during training`, beginning with an initial small batch size that increases between selected epochs.
- This approach delivers the accuracy of training with small batch sizes, while improving performance during later epochs through the use of  progressively larger batch sizes.
- The paper makes this adaptive change in such a way that 'learning_rate/batch_size' stays a constant.
- Paper implementation
  - momentum = 0.9
  - init_batch_size = 128
  - init_lr = 0.1
  - After 20 epochs, lr decays to half and batch_size is doubled.
- The study shows that increasing the batch size too much and too early during training can lead to poor convergence.