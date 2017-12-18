%%
clear

%%
sample_number = 12000;


%%
data = dlmread('lbp.txt', ' ', [0,0,sample_number-1,2477]);

mean_X = mean(data);

[~,egenVect,egenVal] = princomp(data);

sum = sum(egenVal);
summer = 0;

for i = 1:numel(egenVal)
    summer = summer + egenVal(i);
    if summer/sum >= 0.99
        cutoff = i;
        break
    end
end

if cutoff >= 400
    cutoff = 400;
end

egenVect_X = egenVect(1:cutoff,:);

save('pca_lbp.mat', 'mean_X', 'cutoff', 'egenVect_X');

fId = fopen('PCA_Feature.txt', 'w');

fprintf(fId, '%.4f ', mean_X);
fprintf(fId, '\r\n');
fprintf(fId, '%d\r\n', cutoff);
for i = 1:size(egenVect_X,1)
   fprintf(fId, '%d ', int16(egenVect_X(i,:)));
   fprintf(fId, '\r\n');
end

fclose(fId);

%%
