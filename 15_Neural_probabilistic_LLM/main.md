# A Neural Probabilistic Language Model
### Yoshua Bengio, 2003

- The paper attempt to create a  model learns simultaneously `(1) a distributed representation for each word along with (2) the probability function for word sequences,`expressed in terms of these representations.
- Statistical methods , like n-gram, work fine. But the model gets more complex as vocabulory increases. It also doesnt take into account similarity of words, when predicting.
- We can reduce the curse of dimensionality by using distributional representation(embeddings)

### The proposed approach
1. associate with each word in the vocabulary a distributed word feature vector (a real-valued vector in Rm ),
2. express the joint probability function of word sequences in terms of the feature vectors of these words in the sequence, and
3. learn simultaneously the word feature vectors and the parameters of that probability function.

We use embedding to represent words in a lower dimension. In the proposed model, it will so generalize because “similar” words are expected to have a similar feature vector, and because the probability function is a smooth function of these feature values, a small change in the features will induce a small change in the probability.