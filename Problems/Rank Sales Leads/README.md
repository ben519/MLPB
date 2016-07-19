# Rank Sales Leads
Given a set of sales leads (i.e. prospective customers), rank which ones will most likely convert to a sale.

### Hypothetical Use Case
Suppose you sell software that helps stores manage their inventory. You collect leads on thousands of potential customers, and your strategy is to cold-call them and pitch your product. You can only make 100 phone calls per day, so you want to identify leads with a high probability of converting to a sale. By calling leads randomly, you only generate about 2 sales per day (a 2% hit ratio).

In this case, it's important to identify the leads which are most likely to convert to a sale. We are **not** interested interested in optimizing accuracy rate, because it's likely that no leads have a > 50% chance of becoming a sale, in which case the most accurate model will be the one that predicts no lead will convert to a sale. A better objective is to predict the probability that each lead becomes a sale... but even that's not necessary. In this scenario, we more intested in ranking the likelihood that each lead becomes a sale. With this in mind, area under the ROC curve (AUC ROC) is a good and typical candidate objective function. An even better objective function might be Partial AUC ROC, only considering the highest 10% or 20% of leads predicted to convert, in order to specifically reduce our model's false positive rate.

### Tags
[binary-classification] [classification] [gradient_boosting] [logistic-regression] [one-hot-encoding] [python] [R] [random-forest] [rank-target] [sparse-data] [supervised-learning] [xgboost]
