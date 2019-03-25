%% Clear Workspace
clear all

%% Read Training Data CSV
training_dataset_name = 'TestFinalMax.csv';
train = csvread(training_dataset_name);

%% Read Testing Data CSV
testing_dataset_filename = 'TrainingFinalMax.csv';
test = csvread(testing_dataset_filename);

%% Select Features(x) and Labels(y) from test dataset
%l = 1;
%u = 1000; % Increase this to get better results but more computation time

%% Select Features(x) and Labels(y) from test dataset
xt = test(:,1:54);
yt = test(:,end);
total=size(test,1);

%% Select Features(x) and Labels(y) from training dataset
x = train(:,1:54);
y = train(:,end);
j=10;
%% Create kNN Model object
for k=1:25
Mdl= fitcknn(x,y,'NumNeighbors',k,'Standardize',1);
%knnModel(:,:,k)=Mdl;


%% Calculate misclassification error
cvmodel = crossval(Mdl,'kfold',j);
cvError(k,1) = kfoldLoss(cvmodel);


%% Predict labels using Model
y_predicted = predict(Mdl,xt);

%% Calculate Confusion Matrix
[R,ord] = confusionmat(yt,y_predicted,'order',[1 2 3 4 5 6 7]);

%% Display Cross Validation Error
display(['Cross Validation Error for k = ' num2str(k) ' and ' num2str(j) '-fold =' num2str(cvError(k,1)*100) '%']);

%% Display Confusion Matrix
%display(R);
%% Calculating Accuracy
Accuracy(k,1)=(sum(y_predicted==yt)/total)*100;

%%
clear Mdl;
clear cvmodel;
end
%%
plot(cvError*100);
title('Cross Validation Error');
xlabel('k');
ylabel('Error Percentage');
plot(Accuracy);
title('KNN Accuracy');
xlabel('k');
ylabel('Accuracy');

%legend('2-fold','3-fold','4-fold','5-fold','6-fold','7-fold','8-fold','9-fold','10-fold');