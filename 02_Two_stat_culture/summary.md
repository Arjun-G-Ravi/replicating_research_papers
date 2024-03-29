# Statistical Modeling: The Two Cultures, 2001
#### - Leo Breiman

```
Abstract. There are two cultures in the use of statistical modeling to reach conclusions from data. One assumes that the data are generated by a given stochastic data model. The other uses algorithmic models and treats the data mechanism as unknown. The statistical community has been committed to the almost exclusive use of data models. This commitment has led to irrelevant theory, questionable conclusions, and has kept statisticians from working on a large range of interesting current problems. Algorithmic modeling, both in theory and practice, has developed rapidly in ﬁelds outside statistics. It can be used both on large complex data sets and as a more accurate and informative alternative to data modeling on smaller data sets. If our goal as a ﬁeld is to use data to solve problems, then we need to move away from exclusive dependence on data models and adopt a more diverse set of tools.
```

There are two reasons for performing statistics:
1. To be able to predict future data
2. To learn something valuable about the current data

Data Modeling Cultures:

1. Emphasis on Inference: 
- This culture focuses on building models that represent the `underlying data-generating process`. It emphasizes parameter estimation and hypothesis testing for making inferences about the population.
- Assumptions and Simplifications: Data modeling `often relies on assumptions` about the distribution of the data and the relationship between variables. The models are typically designed with a set of assumptions that may not always hold in real-world situations.
- Uses models that makes assumptions on the datasets. Eg: Linear Regression, Logistic regression

2. Algorithmic Modeling Culture:
- Algorithmic modeling `prioritizes predictive accuracy over detailed understanding of the underlying data structure`. The focus is on building models that make accurate predictions without necessarily capturing the true underlying process.
- Flexibility and Complexity: Algorithmic models, such as machine learning algorithms, can handle complex relationships in the data and are often more flexible. They are less concerned with assumptions and can adapt to different types of data.
- Uses models that try to learn completely from the stats. Eg: NN, Decision trees

The author claims that most of the statisticians fall in the first category, and make wrong assumptions to study the data, which can end up affecting the quality of the study. A better way to look into things is by giving more importance to the second culture, and thus finding a balance between the two cultures.