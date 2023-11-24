# MentalHealth - ANN

### Abstract

Amidst a rising tide of mental health issues, there is a clear call for a transformative approach in monitoring, aimed at better understanding and skillfully navigating this intricate landscape. In this paper, we explore the development of an Artificial Neural Network (ANN) for predicting anxiety and depression using an existing dataset. This dataset tracked mental health symptoms by week for a total of 10 weeks during the initial wave of COVID-19 (April  to June, 2020) in a geographically diverse sample of adult participants in the United States. This timeframe, marked by a surge in mental health disorders, provides a vital context for our analysis. Bridging this period of heightened mental health challenges with our research aims, the study focuses on extracting patterns of anxiety and depression using an ANN through qualitative attributes of individuals. 
% By doing so, it seeks to unravel how these conditions evolve under extraordinary circumstances, such as a global pandemic. 
The goal is to enhance understanding of mental health dynamics in crisis situations, using the predictive power of ANNs to analyze complex psychological data and extend to other techniques such as time-series analysis.


### Methodology

This section outlines the process of constructing an ANN model to predict mental health outcomes from questionnaire data. The data preprocessing involved selecting numerical data, which encompassed mental health scores for depression (sds-score) and anxiety (stai-s-score), and standardizing this data across ten datasets after a rigorous reformatting process. 

Our ANN model's structure was built with an input layer, two hidden layers each containing 100 neurons, and an output layer (predicting sds-score and stai-s-score), all optimized using the ReLU activation function and the Adam optimization algorithm. The robustness of the model was ensured through k-fold cross-validation, with k set to 5, to optimize model performance while ensuring its generalizability. This preparation led to a well-tuned ANN model that can accurately predict mental health conditions, striking a balance between complexity and predictive reliability.


### Results:

The study's results section reports that the ANN model effectively discerned trends in mental health scores across various demographics and pandemic phases. Particularly, higher depression and anxiety levels were associated with all surveyed participants, with notable distinctions based on socioeconomic variables such as income and education. The model's predictive performance, evaluated using Mean Squared Error (MSE) across five rounds of cross-validation, yielded an average MSE of 0.530, indicating a high level of accuracy in predicting mental health scores. The MSE values ranged narrowly from 0.5064 to 0.5393, suggesting consistency in the model's performance. Furthermore, the model's predictions were closely aligned with the actual scores, deviating by only 4.8 units on average from the true values, thus highlighting the model's potential utility in clinical settings. Despite the limitations posed by the dataset's scope, these results underscore the model's robustness and the efficacy of using ANN for mental health assessments.


---
### Installation:

1. Clone Repo

2. pip install -r requirements.txt

3. Run python file in the main folder

### License: