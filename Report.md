# Predicting Amazon Movie Review Star Ratings

## Introduction

In this project, I aimed to predict star ratings for Amazon movie reviews using the provided dataset. 
## Final Algorithm Implemented

I developed a stacking ensemble model that combines several classifiers to enhance overall performance. Here's a breakdown of the models I used:

1. **Multinomial Naive Bayes (`MultinomialNB`)

2. **Linear Support Vector Classifier (`LinearSVC`)

3. **Decision Tree Classifier (`DecisionTreeClassifier`)

4. **Logistic Regression (`LogisticRegression`)

5. **Random Forest Classifier (`RandomForestClassifier`)

6. **XGBoost Classifier (`xgb.XGBClassifier`)

7. **Gradient Boosting Classifier (`GradientBoostingClassifier`)

The predictions from these base models were stacked and used as input features for the Gradient Boosting Classifier, which made the final predictions.

### Feature Engineering

**Patterns Noticed:**

- **Textual Sentiment Indicators:** Positive reviews often use words like "great," "amazing," and "excellent," while negative reviews use words like "bad," "worst," and "terrible."
  
- **Review Length:** Extremely positive or negative reviews tend to be longer, possibly because users feel more compelled to explain their strong feelings.

- **Helpfulness Votes:** Reviews marked as helpful by other users might be more reflective of the true quality of the product.

**How I Used These Patterns:**

- **Combined Text Feature:** Merged `Summary` and `Text` into a single `Combined_Text` feature to capture all textual information.

- **Text Preprocessing:** Converted text to lowercase, removed punctuation and numbers, and applied stemming using NLTK's Snowball Stemmer. Removed stopwords to reduce noise.

- **TF-IDF Vectorization with N-grams:** Used `TfidfVectorizer` with up to trigrams to capture phrases indicative of certain sentiments.

- **Feature Selection with Chi-Squared Test:** Selected the top 5,000 textual features that are most correlated with the target variable.

- **Dimensionality Reduction with SVD:** Applied Truncated SVD to reduce dimensionality while preserving essential information.

- **Engineered Numeric Features:**

  - **Helpfulness Ratio:** Calculated as `HelpfulnessNumerator` divided by `HelpfulnessDenominator`.

  - **Review Length Metrics:** Included `Review_Length`, `Word_Count`, and `Char_Count` as features.

  - **Average Scores:** Computed average product and user scores (`Avg_Product_Score`, `Avg_User_Score`) to capture inherent biases.

  - **Temporal Feature:** Extracted `Review_Year` from the timestamp to account for temporal trends.

### Handling Class Imbalance

**Issue Identified:**

- The dataset is heavily imbalanced, with the majority of reviews having a score of 5.

**Solution Implemented:**

- **Class Weights:** Calculated class weights using `compute_class_weight` and applied them to models that support this parameter.

- **Impact:** This approach gave more importance to minority classes, improving the model's ability to predict lower-frequency ratings.

### Ensemble and Stacking

**Rationale:**

- Combining multiple models can capture different aspects of the data, leading to better overall performance.

**Implementation:**

- **Base Models:** Each model makes predictions on the validation set.

- **Stacking:** Collected predictions from base models and used them as features for the meta-model.

- **Meta-Model:** Trained a `GradientBoostingClassifier` on these stacked features.

### Offline Evaluation

**Approach:**

- Split the data into training and validation sets using stratified sampling to maintain class proportions.

- Used metrics like accuracy, precision, recall, and the confusion matrix to evaluate model performance.

- Performed cross-validation to ensure the model generalizes well to unseen data.

## Thought Process and Understanding of Concepts

Throughout the project, I tried to apply concepts from class and build upon them:

- **Support Vector Machines and Naive Bayes:** Leveraged these algorithms for text classification, understanding their strengths in handling high-dimensional data.

- **Decision Trees:** Used for their interpretability and ability to model non-linear relationships.

- **Latent Semantic Analysis and SVD:** Applied SVD for dimensionality reduction, a concept we explored in class.

- **Random Forest Classifier:** Random Forests can capture complex interactions and are less prone to overfitting compared to single decision trees. I Included it as one of the base models in the ensemble.

- **Feature Engineering:** Recognized the importance of intelligent feature construction, such as combining textual and numeric data, to enhance model performance.

- **Class Imbalance Handling:** Applied class weighting, a technique discussed in the context of imbalanced datasets.

## External Methods Used

### Logistic Regression

- **Why I Used It:** Logistic Regression is a straightforward and interpretable model suitable for multi-class classification.

- **How I Used It:** Trained it on combined features to serve as a strong baseline and contribute to the ensemble.

- **Reference:** Scikit-learn documentation on Logistic Regression [1].

### XGBoost Classifier

- **Why I Used It:** Known for its high performance and efficiency, XGBoost can handle large datasets and complex patterns.

- **How I Used It:** Used as a base model, expecting it to improve the ensemble's predictive power.

- **Reference:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System* [3].

### Gradient Boosting Classifier

- **Why I Used It:** As a meta-model, it can effectively learn from the base models' predictions.

- **How I Used It:** Trained on stacked predictions to make the final classification.

- **Reference:** Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine* [4].


---

**References:**

[1] Scikit-learn: Logistic Regression. Available at: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

[3] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794. DOI: 10.1145/2939672.2939785

[4] Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232. DOI: 10.1214/aos/1013203451

---
