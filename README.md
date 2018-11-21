# OCR-System-Using-a-Linear-Perception

# Training part

# wts = trainGradientDescent(X,Y,learningRate,momentum)
The above function returns the weights when given the training data X, target values Y and the learning rate and momentum.  

# Test part

A test function for getting predictions
# predictions = testGradientDescent(testX,regressionCoefficients) 


# Evaluation part
Evaluation function that makes a confusion matrix
 [TP FP TN FN] = confusionMatrix(prediction,actualLabels)
The confusion matrix tells us which class is classified correctly and how many mistakes we are making:

Compute the following: 
BalancedAccuracyRate (BAC) = (TP/totalPositiveLabels+TN/totalNegativeLabels)/2

# Main script: Bring it all together

a. Reads the training data and builds a perceptron model.  Next it gets predictions from the model using the training set as well as the test set.

b. Mapping the OCR labels to the predictions.  So for example if your prediction is 10, then how will you map it to a label?

c. For the training data as well as the test data, make the confusion matrix for different values of lambda, as given in the report.
