function accuracy= measure_accuracy(inputClass,outputClass)

% inputClass contains the final class of the testing data given by user
% outputClass contains the final class of the testing data predicted by
% algoritm

% Count will store the number of times the correct value is predicted
count=0;

% Matching the actual outcome with the predicted output of the ALgorithm
numberofInstances=size(inputClass,1);
for i=1:numberofInstances
    if(inputClass(i,1)==outputClass(i,1))
     count=count+1;         
    end
end

% Now finding the accuracy of the algorithm
  accuracy= ((count/numberofInstances)*100);
  
  fprintf('The accuracy of prediction is: %f%%\n',accuracy);
