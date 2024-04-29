clear
clc

datapath = '/home/ruohan/ScientificProject/UKBdata_analysis/';
load(fullfile(datapath,'data','UKBdata_HCPMMP_SubMeanFC_left_somatosensory.mat')); % mean of FCs within the somatosensory/motor cortex in the left hemisphere
load(fullfile(datapath,'data','UKBdata_select_behaviour.mat')); % physical characteristics data
load(fullfile(datapath,'data','UKB_covariates.mat')); % set covariates
[cov_behav_SubID,ind_cov,ind_behav] = intersect(part_cov.SubID,selected_behaviour(:,1));
cov = [part_cov.Age part_cov.BMI part_cov.Qualifications part_cov.Townsend_index part_cov.Smoking_status ...
    part_cov.Drinker_status part_cov.Head_motion part_cov.Site1 part_cov.Site2];
cov = cov(ind_cov,:);
Field6032 = selected_behaviour(ind_behav,4); % maximum workload during fitness test
Field23100 = selected_behaviour(ind_behav,6); % whole body fat mass

%% set FC matrices
[~,ind_FC,ind_cov_behav] = intersect(FC_SubID,cov_behav_SubID);
FC = sub_mean_FC(ind_FC,:);
cov = cov(ind_cov_behav,:);
Field6032 = Field6032(ind_cov_behav,:);
Field23100 = Field23100(ind_cov_behav,:);

%% Correlation analysis between the FCs and the selected Field.
[r_Field6032,p_Field6032] = partialcorr(FC,Field6032,cov,'Rows','complete');
[r_Field23100,p_Field23100] = partialcorr(FC,Field23100,cov,'Rows','complete');
