

%% Read CSV

w = csvread('TrainingFinal_1.csv');

%%
d_max = w(:,1:54);
%d_mean = w(:,1:54);
%% 
w_max = max(w(:,1:54));
%w_mean = mean(w(:,1:54));
%%
d_max = d_max ./ repmat(w_max,size(w,1),1);
%d_mean = d_mean ./ repmat(w_mean,size(w,1),1);

y=w(:,55);
%%
dat_max = [d_max y];
%dat_mean = [d_mean  y];
% %%
% st_bin = w(:,15:54);
% [r,c] = find(st_bin==1);
% st(r,1) = c;
% 
% wa_bin = w(:,11:14);
% [r,c] = find(wa_bin == 1);
% wa(r,1) = c;
% 
% %%
% d_max = [w(:,1:10) wa st];
% d_mean = [w(:,1:10) wa st];
% 
% y=w(:,55);
% 
% %% 
% w_max = max([w(:,1:10) wa st]);
% w_mean = mean([w(:,1:10) wa st]);
% 
% %%
% d_max = d_max ./ repmat(w_max,size(w,1),1);
% d_mean = d_mean ./ repmat(w_mean,size(w,1),1);
% 
% %%
% dat_max = [d_max y];
% dat_mean = [d_mean  y];

%%
csvwrite('TestFinalNorm.csv',dat_max);
%csvwrite('covtype_all_mean.csv',dat_mean);