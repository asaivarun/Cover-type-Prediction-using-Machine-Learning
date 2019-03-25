function func= naiveBayes_ml(inputData,sample)

% Extracting all the columns except the Class 
  data=inputData(:,1:(size(inputData,2)-1));

% Extracting the class column of the data for calculating the Priori(Class
% probability)

% Extracting the classes from the data
  classes=inputData(:,size(inputData,2));

% Using the function unique()to find the classes in the training data
  final_class=unique(classes); 

% Here a matrix is created to store the class - frequency- probability
    % 1.0000  147.0000    0.2639
    % 2.0000  330.0000    0.5925
    % 5.0000   80.0000    0.1436
  priori_data=[final_class zeros(size(final_class,1),1) zeros(size(final_class,1),1)];

% Frequency of each class in training data by comparing each class with the
% classes in traing data
 for i=1:size(classes,1)
     for j=1:size(final_class,1)
        if(classes(i)== final_class(j))
           priori_data(j,2)=priori_data(j,2)+1;
 % Computing the probability of each class in the data(p(y))  
           priori_data(j,3)=priori_data(j,2)/size(classes,1);
        end
     end
 end

% Here a Matrix is craeted which stores the class contional probability of
% the given data
 %   1     0     0     0     0     0     0     0     0     0     0     0     0
 %   2     0     0     0     0     0     0     0     0     0     0     0     0
 %   5     0     0     0     0     0     0     0     0     0     0     0     0
  classConditionalData=[final_class zeros(size(final_class,1),(size(data,2)))];

% Sorting the inputData on the  basis of class 
  sortrows(inputData,size(inputData,2));

% Now finding the Class-conditional probablity of the given data(p(x(i)/y(q)))
 for i=1:size(data,1)
     for attribute=1:size(data,2)
        for k=1:size(final_class,1)
            if((sample(1,attribute)==inputData(i,attribute))&& (inputData(i,size(inputData,2))==final_class(k,1)))
                 classConditionalData(k,attribute+1)=((classConditionalData(k,attribute+1)+1)/priori_data(k,2));         
            end
        end
     end
 end
 
 % Now finding the (p(x/y(q))) by multiplying all the p(p(x(i)/y(j)))
       classConditionalProbabilities=prod(classConditionalData(:,2:size(classConditionalData,2)),2);
 
 %  Now finding the p(y(q))*(p(x/y(q))) and computing that which class has
 %  highest value
       classConditionalProbabilities=classConditionalProbabilities.*priori_data(:,3);
  
 % Finding the class with the highest probability and hence is the
 % predictio of the algorithm
    [x,max_index]=max( classConditionalProbabilities);
 
  fprintf('The class %d has maximum probability of %f.\n',priori_data(max_index,1),x);  
end
