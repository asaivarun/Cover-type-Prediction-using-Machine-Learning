%% Convert the output classes into the encoded form, store it in a file

% Read the data from the file 
outputFile=csvread('TestFinalNorm.csv');
% Extract the output class from the above data matrix
outputClass=outputFile(:,end);

% Create a Zeros matrix to store the encoded class
encoded=zeros(size(outputClass,1),7);

for i=1:size(outputClass,1)
   % For class 1
    if(outputClass(i,1)==1)
       encoded(i,:)=[1 0 0 0 0 0 0];
    end
   % For class 2
   if(outputClass(i,1)==2)
       encoded(i,:)=[0 1 0 0 0 0 0];
   end
   % For class 3
   if(outputClass(i,1)==3)
       encoded(i,:)=[0 0 1 0 0 0 0];
   end
   % For class 4
   if(outputClass(i,1)==4)
       encoded(i,:)=[0 0 0 1 0 0 0];
   end
   % For class 5
   if(outputClass(i,1)==5)
       encoded(i,:)=[0 0 0 0 1 0 0];
   end
   % For class 6
   if(outputClass(i,1)==6)
       encoded(i,:)=[0 0 0 0 0 1 0];
   end
   % For class 7
   if(outputClass(i,1)==7)
       encoded(i,:)=[0 0 0 0 0 0 1];
   end
end

% Now write the csv file of the encoded output class
csvwrite('TestEncoded.csv',encoded);