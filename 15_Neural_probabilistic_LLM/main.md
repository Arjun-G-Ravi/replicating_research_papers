# A Neural Probabilistic Language Model
### Yoshua Bengio, 2003

- The paper attempt to create a  model learns simultaneously `(1) a distributed representation for each word along with (2) the probability function for word sequences,`expressed in terms of these representations.
- Statistical methods , like n-gram, work fine. But the model gets more complex as vocabulory increases. It also doesnt take into account similarity of words, when predicting.
- We can reduce the curse of dimensionality by using distributional representation(embeddings)