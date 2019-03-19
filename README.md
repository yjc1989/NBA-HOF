# Machine Learning Applications on NBA Players' Statistics - What Matters Most to the Hall of Fame Committees?

##### Vickram N. Premakumar & Yuan-Chi Yang

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/neural-network-pca/two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

## Introduction

The National Basketball Association (NBA) is the most prominent professional basketball league in the world. Earning a spot on an NBA team's roster is the aspiration of thousands of young players both in the United States and abroad. Among these great players, a tiny percentage of them are enshrined in the Naismith Memorial Basketball Hall of Fame (HOF), the highest career-spanning honor that can be awarded. 

In this work, we study the relationship between the player's career statistics and whether the player is inducted in the HOF. Our focus is how a player is inducted into the HOF. Specifically we want to make actionable recommendations for a player whose career goal is to be a Hall of Famer. The approach such a player should take is very opaque - a total of seven committees are formed to screen and select inductees and the selection criteria for the HOF are not made widely known. Career-wide basketball statistics should be involved in the decision making; however, these are rarely cited as reasons for induction. Instead, specific games and historic moments are used as examples of a Hall of Famer's greatness. This reflects the complex decision making process that goes into HOF selection. Therefore, machine learning tools are well suited for better understanding this process. With predictive models, we can not only forecast which current players have a good chance of being enshrined in the HOF, but also make recommendations on how to improve their odds. 

In the following, we first discuss how we gather and prepare clean data for analysis. We then present the analysis, specifically on how we select the most prominent features and how we apply machine-learning models.

## Data Pipeline

In this section, we discuss our data pipeline. In particular, how we gather, cleanse, and prepare the data for analysis.

### Data gathering

We gather data from various sources online. First we download the season-by-season player statistics data from kaggle. We then scrape up-to-date lists of Hall of Fame inductees and other seasonal awards (MVPs, All-Star Selections, Championships, etc.) from Wikipedia. For a more detailed discussion, please see the notebook, 'data-gathering.ipynb'. 

### Data Preparation

To prepare clean data, we first tidy the seasonal statistics by deleting entries with duplicated information and replacing missing data by hand where possible, and dropping it otherwise. We then construct the players' career statistics based on the cleansed seasonal statistics. We further append Hall of Fame status and seasonal awards in the data. For a detailed discussion, please see the notebooks, 'data-preparation.ipynb' and 'data-preparation-appending-awards.ipynb'.

## Results

In this section, we present the analysis results. We first discuss how we select the most important features and use logistic regression as a tool of validation. We also develop predictive models based on logistic regression and other machine learning techniques and show their corresponding results. 

In order to achieve uniformity in our training dataset, we use players whose statistics are well-recorded (careers started after 1982) and who are eligible for HOF selection (careers ended before 2015). We then search for the optimal model to predict if a player is a Hall of Famer. We then apply the trained model to players who are not yet eligible to make predictions.

### Feature Selection and Logistic Regression

First, we find that the features can be grouped based on correlations, for example, number of game played, 'G,' and number of starting game, 'GS,' are highly correlated and can be treated as one group. Then for each group, we find the representing features based on the correlations with the Hall of Fame status, 'HOF.' We then sort the representing features based on the correlations with 'HOF,' which corresponding to the order of importance. The top five representing features are 'AllStar' (number of all-star games attended), 'VORP' (value over replacement player), 'WS' (win share), 'AllStar_MVP' (number of all-star MVP awarded), and 'First team' (number of All-NBA first team selections).
    
|Representing Feature|Correlation with 'HOF'|
|---|---|
|'AllStar'|0.836601|
|'VORP'|0.616421|
|'WS'|0.553751|
|'AllStar_MVP'|0.506579|
|'First team'| 0.436193|

This table shows that the number of all-star selections, 'AllStar,' has a very high correlation with 'HOF' comparing to other representing features.

#### Validation Through logistic regression

We then explore the performance of logistic repression models by using the representing features. Specifically, we search for the optimal model based on learning curves for selected features, and cross-validation curves over the number of representing features.

We first study the learning curves for the logistic regression models using only the most important representing feature, 'Allstar,' and two most important representing features, 'AllStar' and 'VORP.' Monitoring F1 score, we find that the model with one representing feature has low bias and low variance, with training score saturating at around 0.88 and test score at around 0.87. The model with two representing features has lower bias but higher variance, with training score saturating at around 0.895 and test score at around 0.87. This difference in the two models suggests that the model with two representing features overfits the data in comparison to the model with only one representing feature.

Learning Curves

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/learn-curve-combine.png?raw=true" alt="Centered Image"/>
</p>

We next plot the cross-validation curve over the number of representing features, aiming at finding the optimal number of representing features. As expected, the optimal model is using only one representing feature, 'AllStar.'

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/cross-validation-number-of-features.png?raw=true" alt="Centered Image"/>
</p>

### Logistic Regression Results

We now present the results for logistic regression models using only one representing feature, 'AllStar,' and two representing features, 'AllStar' and 'VORP.'

####  Logistic Regression Model Using 'AllStar'

We first plot the distribution of 'AllStar,' grouping players by Hall of Fame status. According to the figure, a good boundary could be between 'AllStar'=5 and 'AllStar'=6. We also note that the Hall of Famer who has never attended any all-star games is Arvydas Sabonis, who was inducted in recognition of his achievement in international competition rather than outstanding NBA career. As a outlier, we later on drop this data point.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/distribution-one-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

Training the model using the whole working data set makes the following predictions on the probability of being inducted based on the number of all-star games attended.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/probability-one-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

This basically predicts that players who have attended at least six all-star games will be inducted.
The training performance and the prediction on players who are not yet eligible for HOF selection is presented later, along with the analogous results for the model using two representing features.

####  Logistic Regression Model Using 'AllStar' and 'VORP'

We now plot the distribution of 'AllStar' and 'VORP,' grouping players according to HOF status. According to the figure, a line connecting ('AllStar','VORP') = (5,60) and (6,0) seems to be a good boundary. We again note that Arvydas Sabonis is an outlier and will be droped for model training.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/distribution-two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

Training the data using the whole working data set determine the following boundary line:

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/classification-two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>


#### Results

We now compare the performance of the two models by using precision, recall, and F1 score.

|Model|Precision|Recall|F1|
|---|---|---|---|
|'AllStar'|0.926|0.862|0.893|
|'AllStar' and 'VORP'|0.929|0.897|0.912|

This table shows that the perfomance of the model using two representing features is slightly better.

We also explore the quality of the fitting by using the package 'statsmodels.' For the model using only 'AllStar,'

|Feature|Coefficient|p-value|
|---|---|---|
|'AllStar|1.948|0.000|

For the model using 'AllStar' and 'VORP,'

|Feature|Coefficient|p-value|
|---|---|---|
|'AllStar|1.804|0.000|
|'VORP'|0.352|0.275|

Here, 'VORP' doesn't reach statistic significance. This suggests that the model using the two representing features might overfit the data. Though increasing number of data might improve the results, gathering more data is slow, limited by the number of NBA players each season. 

We now show predictions on the players who are not yet eligible, assuming their careers end in 2018. Specifically, we show the players' names and the probabilities of being inducted by the two models, and sort the results by the probability predicted by models with two features:

|Player|Probability using 'AllStar'|Probability using 'AllStar' and 'VORP'|
|---|---|---|
|Kobe Bryant|0.999944|0.999943|
|Kevin Garnett|0.999419|0.999658|
|Tim Duncan|0.999419|0.999628|
|LeBron James|0.998733|0.999521|
|Dirk Nowitzki|0.997238|0.997610|
|Dwyane Wade|0.993990|0.994365|
|Chris Bosh|0.986971|0.980391|
|Paul Pierce|0.971987|0.977831|
|Chris Paul|0.940801|0.962368|
|Carmelo Anthony|0.971987|0.961662|
|Kevin Durant|0.940801|0.943087|
|Vince Carter|0.879214|0.902361|
|Dwight Howard|0.879214|0.874906|
|Russell Westbrook|0.769268|0.800773|
|Joe Johnson|0.769268|0.732703|
|Pau Gasol|0.604286|0.683180|
|Stephen Curry|0.604286|0.633659|
|James Harden|0.604286|0.633249|
|Tony Parker|0.604286|0.572100|
|LaMarcus Aldridge|0.604286|0.551259|
|Paul George|0.604286|0.546018|
|Amar'e Stoudemire|0.604286|0.526281|
|Kyrie Irving|0.604286|0.524521|

Though the predicted probabilities for the two models on the same player have slightly different values, there is no player who is predicted to be inducted by only one model. That is, the two predicted class boundary classify these players in the same way.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/feature-selection-logistic-regression/prediction-two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

Here, the boundary between the brown (predicted HOFer) and light blue (predicted not HOFer) is the class boundary predicted by the model using 'AllStar' and 'VORP,' and the gray dashed line is the class boundary predicted by the model using only 'AllStar.'

We also explore the players who, based on our impression, have high changes to be inducted into Hall of Fame, but are not predicted with probability over 50%. 

|Player|'AllStar'|'VORP'|Probability using 'AllStar'|Probability using 'AllStar' and 'VORP'|
|---|---|---|---|---|
|Al Horford|5|27.2|0.411572|0.393452|
|Blake Griffin|5|25.6|0.411572|0.386738|
|Kevin Love|5|23.5|0.411572|0.377991|
|John Wall|5|20.3|0.411572|0.364814|
|Anthony Davis|5|17.8|0.411572|0.354658|
|Rajon Rondo|4|21.9|0.242633|0.222777|
|Klay Thompson|4|9.6|0.242633|0.187474|

We believe if these players maintain their current performance for the rest of their career, they are very likely to be inducted into Hall of Fame.


## Other Machine Learning Models

In this section, we present the results using other machine learning models, specifically Naive Bayes and Neural Network. We again explore the model based on only one representing feature, 'AllStar,' and the model using two representing features, 'AllStar' and 'VORP.' 

### Naive Bayes

We use the same training data set as logistic regression, searching for the optimal model to predict if the player is a Hall of Famer. We then apply the trained model to players who are not yet eligible and make predictions based on available statistics to 2018.

####   'AllStar'

Training the data using the whole working data set make the following predition on the probability of being inducted into Hall of Fame based on the number of all-star games attended.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/naive-bayes/probability-one-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

This basically predict that players who have attended at least two all-star games will be inducted. This is very different from logistic regression. The performance is also much worse.

|Model|Precision|Recall|F1|
|---|---|---|---|
|'AllStar'|0.500|0.933|0.651|

Though the recall is high, the class boundary is too far into the 'HOF' = 0 side, rendering low precision.

####   'AllStar' and 'VORP'

Training the data using the whole working data set determine the following boundary line:

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/naive-bayes/classification-two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

This figure also shows that the class boundary is too far into the 'HOF' = 0 region, resulting in low precision but high recall. Hence, the performance is also much worse.

|Model|Precision|Recall|F1|
|---|---|---|---|
|'AllStar' and 'VORP'|0.363|0.967|0.528|


### Neural Network

To replicate the complex decision making that goes on between the many humans and committees that are responsible for choosing the HOF inductees, we choose to apply a neural network, using all available features in the data set rather than the ones we found to be the most significant.

We construct validation curves for varying depth and sigmoid neurons per layer and find that there is not much improvement in F1 score as we turn up the model complexity. The training F1 score saturates very quickly, which indicates that even the simplest neural network with one layer is fitting the training data very well. Increasing model complexity therefore only provides a moderate increase in generalizability to unknown data.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/neural-network/NNVC.png?raw=true" alt="Centered Image"/>
</p>

This is not a great surprise, we learn from logistic regression that a simple model can have remarkably good performance. A more complex model will potentially suffer from overfitting. Comparing our intuition with the neural network's prediction, we see some heuristic signs of overfitting. Some players in the twilight of their careers, such as Carmelo Anthony, Pau Gasol and Vince Carter, who most NBA enthusiasts consider sure-thing future Hall of Famers, are not predicted to make it, while young talents like Anthony Davis are considered already qualified for HOF status. 

### Principle Component Analysis (PCA)
We perform PCA on our training and test data to identify structure when we group players by HOF status. In order to identify by eye any natural separation of Hall of Famers from other players we project our data onto pairs of principle components. The first five principle components explain 70% of the variance, and we select pairs from this restricted set. The projection into the top two principle components show no signs of separation, so we search through pairings and select the one that best suits our purpose for the following figure.

<p align="center">
  <img src="https://github.com/yjc1989/NBA-HOF/blob/master/figures/neural-network-pca/two-rep-feature.png?raw=true" alt="Centered Image"/>
</p>

Several players in both the training and test data distinguish themselves in a dramatic way - we consider these players to be 'legendary,' achieving singularly fantastic careers.  We label these players on the figure by name. Apart from those players there is no natural splitting of classes using PCA. This, combined with the fact that a simple threshold based solely on the 'AllStar' feature does split the classes up very well indicates to us that the addition of other basketball statistics impedes the performance of a classifier. That is to say, the additional information we get from using all available measures of performance is outweighed by the dilution of structure generated by the high dimensionality of the feature space. Nevertheless, the neural network shows signs of learning the subtleties of the distribution of players by HOF status, and thus we believe that additional training data would improve this model significantly.



# Conclusion and Future Work

We have presented statistical analysis on the dominating factors for a NBA player to be inducted into Hall of Fame, including data gathering from multiple websites, data cleansing, feature selection, training machine learning models, and making predictions. Based on logistic regression, we find that the dominating factor for Hall of Fame induction is the number of all-star games the player has attended, where the player has more than 50% chance if he has attended at least six all-star games. The next most important factor could be 'VORP,' but the model including this feature shows signs of over-fitting. We also select some players who are highly probable according to our impressions but have less than 50% chance to be inducted based on the statistical data until 2018 and the logistic regression models.

The strong correlation between Hall of Fame induction and the number of all-star games attended, though surprising, is actually reasonable. These all-star game players are elected via fan voting or selected by coaches. The all-star game is a showcase or exhibition match and is not played very competitively. Therefore, coaches and fans are incentivized to vote for the most watchable players, not necessarily the most efficient or statistically impressive. The more popular the player is, the more likely he will be voted. While deciding the Hall of Fame inductees, the committees inevitably take popularity into consideration among other basketball statistics. When we look at the principle components of our full data set, we see that the exceedingly simple boundary we identify from feature selection is washed out an there is no simple structure separating the Hall of Famers from the rest of the league. 

Given the significance of all-star game selections, one interesting direction to move forward is to explore dominating factors for this decision-making process. Another interesting question is, what are the factors that can compliment the 'AllStar' feature and can make a more accurate predition on the chances of being a Hall of Famer. Some ideas for this are number of commercials, number of magazine appearances, and whether or not the player has a signature sneaker. 

We also explore different machine learning models, specifically Naive Bayes and Neural Network. We find that Naive Bayes tend to have high recall but low precision, setting the classification boundary too deep into the 'HOF' = 0 regime, while Neural Network tends to overfit the data. The Neural Network gains a small predictive edge over the logistic regression at the cost of model interpretability. We posit that increasing our training data size would allow the neural network to learn more and become a better model for predictions. 

In the future, we would like to apply different machine learning models, such as support vector machines, and compare the results with logistic regression and produce artificial data by executing random rotations in feature space on the existing NBA data.
