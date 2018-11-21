trainX = load('ocr3_7train.txt');   	%loads the training data
trainY = load('ocr3_7trainLabels.txt');   	%loads the training labels

o = ones(size(trainX)(1),1);

trainX = [o trainX]; %adding a column of 1's to training data for bias

learningRate = 0.001;

momentum = 0.9;

wts = trainGradientDescent (trainX,trainY,learningRate,momentum); %training weights using gradient descent


testX = load('ocr3_7test.txt');
testY = load('ocr3_7testLabels.txt');

wts = wts(2:end);

trainX = trainX(:,2:end);

predictions = testGradientDescent(trainX,wts);

predictions = predictions/max(predictions);

predictions(predictions>=2) = 3;

predictions(predictions<2) = 7;

[TP, FP, TN, FN] = confusionMatrix (predictions,trainY);

BAC = (TP/sum(trainY==3)+TN/sum(trainY==7))/2;

predictions_2 = testGradientDescent(testX,wts);

predictions_2 = predictions_2/max(predictions_2);

predictions_2(predictions_2>=2) = 3;

predictions_2(predictions_2<2) = 7;

[TP_2, FP_2, TN_2, FN_2] = confusionMatrix (predictions_2,testY);

BAC_2 = (TP_2/sum(testY==3)+TN_2/sum(testY==7))/2;
