# Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark

Shiv Ram Dubey1, Satish Kumar Singh1, Bidyut Baran Chaudhur

- the main goal of any neural network is to transform
the non-linearly separable input data into more linearly separable abstract features using a hierarchy of layers.
- normalising and standarising is very important for most AFs
- proper initialisation is also important(importance varies according to the AF)

# Activation Functions
![alt text](image.png)

1. Sigmoid
- used in initial days
- faced vanishing gradient problem
- poor convergence
- squashes to (0,1)

1. tanh function
- exhibited zero centric property, better convergence
- squashes to [-1,1]
- Tanh function is computationally inefﬁcient because it involves the computa-
tion of exponential multiple times
 ![alt text](image-1.png)

1. ReLU
- simple, high performance
- relu(x) = max(0,x)
-Various variants
of ReLU have been investigated by tackling its drawbacks, such
as non-utilization of negative values, limited non-linearity and
unbounded output 

4. PReLu
- Parametric ReLU
- the slope for negative input as a trainable parameter
- ![alt text](image-2.png)
- Better than relu in many cases
- However, it can lead to overﬁtting easily which is the down-
side of PReLU.
. 
2. Exponential Unit Based Activation Functions
- proper utilisation of large + and -ve numbers
- The exponential AFs generally lead to the increased non-linearity due to utilization of the negative values.

3. The Exponential Linear Unit (ELU)
- based AF utilizes the negative values with the help of the exponential function.
- ![alt text](image-3.png)
- α is a learnable parameter.

  
1. Swish
   - adaptive function
   - Swish(x) = x*Sigmoid(β*x)
   - Based on the learnt value of β the shape of the Swish AF is adjusted between the linear and ReLU functions.
   - can work on very complex problems

2. GELU
   - Gaussian Error Linear Unit
   - GELU(x) = x * P(X ≤ x).
   - The complexity of GELU increases due to use of probabilistic nature. 

3. PAU
        - Pad´e Activation Unit
        - PAU(x) = P(x)/ Q(x)
        - 

# Output

## Overall
Despite the ReLU being a popular choice, recently proposed AFs such as Swish, Mish, and PAU are also worth trying for different problems.

## Residual connections in network
- The ReLU, LReLU, ELU, GELU, CELU, and PDELUfunctions are better for the networks having residual connections for image classiﬁcation.

## Language Translation
The Tanh and SELU AFs are found better for language translation along with PReLU, LiSHT, SRS and PAU.

## Speech recognition
- It is suggested to use the PReLU, GELU, Swish, Mish and PAU activation function for speech recognition.

## Small datasets
- ReLU generally outperforms more complex activation functions
  - 