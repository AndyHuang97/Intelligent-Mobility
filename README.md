# Intelligent-Mobility
## TODO
Classification task: 
- use all target values
- merge target values for yes and no
- delete maybe yes or not
## 1. Data Exploration - Interaction Effects
### 1.1. Guiding Principles in Search of Interactions
Principles for identifying significant predictive interactions, by Wu and Hamada ([2011](https://bookdown.org/max/FES/references.html#ref-wu2011experiments)):
1. **interaction hierarchy** principle: higher degree of the interaction, the less likely the interaction will explain variation in the response
2. **effect sparsity** principle: only a fraction of the possible effects truly explain a significant amount of response variation
3. **effect heredity** principle:  interaction terms may only be considered if the ordered terms preceding the interaction are effective at explaining response variation
   - *strong* heredity: all lower-level preceding terms must explain a significant amount of response variation
   - *weak* heredity: consider any possible interaction with the significant factor
### 1.2. Practical Considerations
1. Does the number of features allow to enumerate all the possible interactions? 
   
   The number of interactions increases quadratically w.r.t. to the number of features
2.  should interaction terms be created before or after preprocessing (centering, scaling, dimension expansion or reduction) the original predictors?

    Interaction terms should probably be created prior to any preprocessing steps. It may also be wise to check the effect of the ordering of these steps.

### 1.3. Brute-Force Approach
#### 1.3.1 Simple Screening
**Note**: use linear regression for continuous response and logistic regression for categorical response

The traditional approach to screening for important interaction terms is to use nested statistical models. For a linear regression model with two predictors, $x_1$ and $x_2$, the main effects model is:
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + error$$
The second model with main effects plus an interaction is:
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_1x_2 + error$$

- These two models are called “nested”" since the first model is a subset of the second. When models are nested, a statistical comparison can be made regarding the amount of additional information that is captured by the interaction term.
-  For linear regression, the residual error is compared between these two models and the hypothesis test evaluates whether the improvement in error, adjusted for degrees of freedom, is sufficient to be considered real
-  The statistical test results in a p-value which reflects the probability that the additional information captured by the term is due to random chance. Small p-values, say less than 0.05, would indicate that there is less than a 5% chance that the additional information captured is due to randomness. 5% is the rate of false positive findings, and is a historical rule-of-thumb.
-  For linear regression, the objective function used to compare models is the statistical likelihood (the residual error, in this case). For other models, such as logistic regression, the objective function to compare nested models would be the binomial likelihood.

An alternatively methodology for protecting against false positive findings was to use resampling. It may not be based on a statistical metric, though. (Different objective functions)

Both the traditional approach and the resampling method from Section 2.3 generate p-values. The more comparisons, the higher the chance of finding a false-positive interaction.

There are a range of methods for controlling for false positive findings:
1. One extreme is to not control for false positive finding
2. At the other extreme is the Bonferroni correction
3. False discovery rate (FDR): compromise between no adjustment and the Bonferroni correction.

#### 1.3.2. Penalized Regression
