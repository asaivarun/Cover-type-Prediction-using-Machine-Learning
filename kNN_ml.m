function accuracy= kNN_ml(inputData,sample,k) 
  
  fprintf('Processing.....Please Wait\n');

% Extracting all the columns except the Class 
  data=inputData(:,1:(size(inputData,2)-1));
  
% Extracting the Output Classes of Testing Data for Accuracy Calculation
outputClass=sample(:,end);




% Running the Algoritm for all Test Cases  
  for x=1:size(sample,1)
      
% Extracting the required inputs of the Testing data   
  sample_data=sample(x,1:size(sample,2)-1);

% Evaluating the distance of test data with the training data
% Using Euclidian Distance
   for i=1:size(data,1)
      dis(i)=sum(((data(i,:)-sample_data).^2).^0.5);
   end
% Creating a matrix having distance and final class(for training data)   
    mat=[dis' inputData(:,end)];

% Sorting the distance in ascending order    
    mat=sortrows(mat,1);

% Taking the first k values (having least distance)    
    mat1=mat(1:k,:);
    
% Now when we have the distance of k classes & their respective class, it's
% time to do Majority Voting and find the answer. 
% The answer would be the one with maximum votes, which in this case means
% having the highest "count".

% Majority Voting
% This is done using function sort(A,dim) , where A is matrix & dim is
% column number for which mode is need to be calculated
  final_class(x,1)=mode(mat1(:,2));
  
  % fprintf('%i\n', final_class(x,1));
    
  end 
  
  accuracy=measure_accuracy(final_class,outputClass);
end