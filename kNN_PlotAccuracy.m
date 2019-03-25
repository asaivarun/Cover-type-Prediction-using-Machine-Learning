max =15;
a = zeros(1,max);
for k = 1:max
a(k)=kNN_ml(inputData,sample,k);
end

plot(a);
title('Accuracy');
xlabel('k');
ylabel('Accuracy(in %)');