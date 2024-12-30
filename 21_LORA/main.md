# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

- Edward Hu,  Yelong Shen - Microsoft Corporation

- We propose Low-Rank Adaptation, or LoRA, which freezes the pre-
trained model weights and injects trainable rank decomposition matrices into each
layer of the Transformer architecture, greatly reducing the number of trainable pa-
rameters for downstream tasks.
- LoRA (Low-Rank Adaptation) is a technique designed to efficiently fine-tune large pre-trained models like language models or vision transformers with significantly reduced computational and storage costs.


## Key Concepts:
Parameter Efficiency: LoRA introduces a small set of trainable parameters instead of modifying the entire pre-trained model. This makes fine-tuning feasible even on systems with limited resources.

Low-Rank Decomposition: The idea behind LoRA is to approximate the updates to the model's weights using low-rank matrices. It decomposes the updates into two smaller matrices, which are easier and cheaper to train.

Frozen Pre-trained Model: During LoRA fine-tuning, the original model's parameters remain frozen. Only the new low-rank adaptation parameters are updated.

Efficiency in Multitasking: LoRA is particularly useful for scenarios where a single pre-trained model needs to be adapted to multiple tasks or domains. The low-rank modules can be task-specific and swapped without affecting the main model.

## How It Works:
Weight Decomposition: Assume the weight matrix 𝑊 of a pre-trained model is updated by adding a learned low-rank matrix 
![alt text](image.png)

Then we train only A and B.