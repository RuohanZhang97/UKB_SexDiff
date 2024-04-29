clear
clc

datapath = '/home/ruohan/ScientificProject/UKBdata_analysis/';
load(fullfile(datapath,'data','UKB_covariates.mat')); % set covariates
savefolder = fullfile(datapath,'Results');

%% set FC matrices
load(fullfile(datapath,'data','UKBdata_HCPMMP_FC_.mat')); % FC matrices based on HCP-MMP atlas
num_BrainRegion = 180;
num_Link = num_BrainRegion*(num_BrainRegion-1)/2; % number of links

[FC_cov_SubID,ind_FC,ind_cov] = intersect(FC_SubID,part_cov.SubID);
FC_cov = FC(ind_FC,:,:); % select participants who both have FC matrices and covariates
part_cov = [part_cov.Age part_cov.Sex part_cov.BMI part_cov.Qualifications part_cov.Townsend_index ...
    part_cov.Smoking_status part_cov.Drinker_status part_cov.Head_motion part_cov.Site1 part_cov.Site2];
cov = part_cov(ind_cov,[1 3:end]);
sex = part_cov(ind_cov,2);

clearvars FC;
%% divide dataset into training and test sets 
rng(1);  % For reproducibility
cv1 = cvpartition(sum(sex == 1),'HoldOut',0.2); % positive example: males
cv2 = cvpartition(sum(sex == 0),'HoldOut',0.2); % negative example: females
SVM_Classification.CrossValidation = {cv1,cv2};
ind_pos_test = cv1.test;
ind_neg_test = cv2.test;

%% two sample t-test on sex differences in FCs
ind_female = find(sex == 0); % female
ind_male = find(sex == 1); % male
FC_female = FC_cov(ind_female,:,:);
FC_male = FC_cov(ind_male,:,:);
covariate_female = cov(ind_female,:);
covariate_male = cov(ind_male,:);

clearvars FC_cov;

FC_female_train = FC_female(~ind_neg_test,:,:);
FC_female_test = FC_female(ind_neg_test,:,:);
FC_male_train = FC_male(~ind_pos_test,:,:);
FC_male_test = FC_male(ind_pos_test,:,:);
covariate_female_train = covariate_female(~ind_neg_test,:);
covariate_female_test = covariate_female(ind_neg_test,:);
covariate_male_train = covariate_male(~ind_pos_test,:);
covariate_male_test = covariate_male(ind_pos_test,:);

T_value = nan(num_BrainRegion,num_BrainRegion);
P_value = nan(num_BrainRegion,num_BrainRegion);
for j = 1:num_BrainRegion
    DependentVariable = gretna_fishertrans([FC_male_train(:,1:num_BrainRegion,j); FC_female_train(:,1:num_BrainRegion,j)]); % fisher z transfermation
    GroupLabel = [zeros(size(FC_male_train,1),1); ones(size(FC_female_train,1),1)];
    Covariate = [covariate_male_train; covariate_female_train];
    [T_value(j,:), P_value(j,:)] = ttest2_cov_improve(DependentVariable, GroupLabel, Covariate);
end
Cohen_d_matrix = sqrt(1/size(FC_male_train,1)+1/size(FC_female_train,1)) * T_value;
Bon_threshold = 0.05/num_Link;
upper_d_matrix = triu(Cohen_d_matrix .* (P_value < Bon_threshold),1);
[top_dval,ind_top_dval] = sort(FCtriu2Vector(abs(upper_d_matrix),1),'descend');

%% sex prediction using SVM
FC_train = [reshape(FC_male_train,size(FC_male_train,1),[],1); reshape(FC_female_train,size(FC_female_train,1),[],1)];
FC_test = [reshape(FC_male_test,size(FC_male_test,1),[],1); reshape(FC_female_test,size(FC_female_test,1),[],1)];
train_label = [ones(size(FC_male_train,1),1); zeros(size(FC_female_train,1),1)];
train_label = logical(train_label);
test_label = [ones(size(FC_male_test,1),1); zeros(size(FC_female_test,1),1)];
test_label = logical(test_label);
    
PropThreshold = [0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
for i = 1:length(PropThreshold)
    disp(i)
    prop_threshold = PropThreshold(1,i);
    SVM_result(i).PropThreshold = prop_threshold;
    ind_top_Bon_signif = zeros(num_Link,1);
    ind_top_Bon_signif(ind_top_dval(1:round(sum(top_dval > 0) * prop_threshold))) = 1;
    ind_top_signif_matrix = Vector2FCtriu(ind_top_Bon_signif,1); 
    ind_top_signif_link = reshape(ind_top_signif_matrix,[],1);
    temp_FC_train = FC_train(:,ind_top_signif_link==1);
    temp_FC_test = FC_test(:,ind_top_signif_link==1);

    rng(10);
    model_svm = fitclinear(temp_FC_train,train_label,'OptimizeHyperparameters',{'Lambda','Regularization'},'HyperparameterOptimizationOptions',...
        struct('Kfold',5,'AcquisitionFunctionName','expected-improvement-plus')); % five-fold cross validation on training set
    SVM_result(i).Model = model_svm;
    [label_svm,score_svm] = predict(model_svm,temp_FC_test);
    SVM_result(i).Predicted_label = label_svm;
    SVM_result(i).Predicted_score = score_svm;
    SVM_result(i).Predicted_accuracy = sum(label_svm == test_label)/length(test_label);
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(test_label,score_svm(:,2),'true');
    SVM_result(i).ROC_X = Xsvm;
    SVM_result(i).ROC_Y = Ysvm;
    SVM_result(i).ROC_T = Tsvm;
    SVM_result(i).AUC = AUCsvm;
end
SVM_Classification.PredictionResult = SVM_result;

savefile = fullfile(savefolder,'SVM_sex_prediction.mat');
save(savefile,'SVM_Classification');



