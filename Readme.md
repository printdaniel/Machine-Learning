# Machine-Learning
## _The Last Markdown Editor, Ever_
## Prediction Method:
1 Pass our preprocessed data X & y to sklearn to split the data into 4 arrays, where 70% of the data go to training and 30% go to test.
2 Selection of the K-Nearest Neighbour algorighm since our dataset is clustered data.
3 Pass training data to the fit method which do all the training and predictions.
4 Append each prediction accuracy to a list we use in the output.
5 Save the highest and lowest prediction accuracies for the output.
6 Instantiate the prediction of our model to a variable used for the output.
7 Append each concatenaded string of prediction, true label (actual species), prediction accuracy & actual data to a list used for the output.

##Accumulate Accuracies Method:
1 Remove the 2 first and last results of our prediction accuracies to eleminate the chance of data noice. (bad data which pollute the predictions and accuracies)
2 Accumulate the remaining accuracies and divide by the frequenzy to get the average accuracy.
-