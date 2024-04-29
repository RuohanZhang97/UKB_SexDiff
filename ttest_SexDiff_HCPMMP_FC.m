clear
clc

datapath = '/home/ruohan/ScientificProject/UKBdata_analysis/';
load(fullfile(datapath,'data','UKB_covariates.mat')); % set covariates
savefolder = fullfile(datapath,'Results');

%% set FC matrices
load(fullfile(datapath,'data','UKBdata_HCPMMP_FC_.mat')); % FC matrices based on HCP-MMP atlas
num_BrainRegion = 360;
num_Link = num_BrainRegion*(num_BrainRegion-1)/2; % number of links
load(fullfile(datapath,'data','HCPMMP_LabelID.mat')); % HCP-MMP region labels
BrainRegionLabel = LabelID(1:num_BrainRegion,4);

[FC_cov_SubID,ind_FC,ind_cov] = intersect(FC_SubID,part_cov.SubID);
FC_cov = FC(ind_FC,:,:); % select participants who both have FC matrices and covariates
part_cov = [part_cov.Age part_cov.Sex part_cov.BMI part_cov.Qualifications part_cov.Townsend_index ...
    part_cov.Smoking_status part_cov.Drinker_status part_cov.Head_motion part_cov.Site1 part_cov.Site2];
cov = part_cov(ind_cov,[1 3:end]);
sex = part_cov(ind_cov,2);

%% two sample t-test on sex differences in FCs
ind_Patient = find(sex == 0); % female
ind_Normal = find(sex == 1); % male
fprintf('Number of female subjects in the analysis is %d.\n', length(ind_Patient));
fprintf('Number of male subjects in the analysis is %d.\n', length(ind_Normal));
FC_Patient = FC_cov(ind_Patient,:,:);
FC_Normal = FC_cov(ind_Normal,:,:);
covariate_pat = cov(ind_Patient,:);
covariate_nor = cov(ind_Normal,:);

T_value = nan(num_BrainRegion,num_BrainRegion);
P_value = nan(num_BrainRegion,num_BrainRegion);
for j = 1:num_BrainRegion
    DependentVariable = gretna_fishertrans([FC_Normal(:,1:num_BrainRegion,j); FC_Patient(:,1:num_BrainRegion,j)]); % fisher z transfermation
    GroupLabel = [zeros(size(FC_Normal,1),1); ones(size(FC_Patient,1),1)];
    Covariate = [covariate_nor; covariate_pat];
    [T_value(j,:), P_value(j,:)] = ttest2_cov_improve(DependentVariable, GroupLabel, Covariate);
end
Cohen_d_matrix = sqrt(1/length(ind_Patient)+1/length(ind_Normal)) * T_value;

%% plot d matrix figure
Bon_threshold = 0.05/num_Link;
upper_d_matrix = triu(Cohen_d_matrix .* (P_value < Bon_threshold),1);
lower_d_matrix = tril(Cohen_d_matrix,-1);

neg_upper_d_matrix = upper_d_matrix .* (upper_d_matrix<0);
pos_upper_d_matrix = upper_d_matrix .* (upper_d_matrix>0);

[top_dval,ind_top_dval] = sort(FCtriu2Vector(abs(neg_upper_d_matrix),1),'descend');
PropThreshold = 0.2;
ind_top_Bon_signif = zeros(num_Link,1);
ind_top_Bon_signif(ind_top_dval(1:round(sum(top_dval > 0)*PropThreshold))) = 1;
ind_top_signif_matrix = Vector2FCtriu(ind_top_Bon_signif,1);
top_signif_upper_d_matrix = neg_upper_d_matrix .* ind_top_signif_matrix + pos_upper_d_matrix; % show top 20% negative links and all positve links
d_matrix = top_signif_upper_d_matrix + lower_d_matrix;
d_matrix(d_matrix == 0) = NaN;

close all;
figure(1)
fig = imagesc(d_matrix);
set(fig,'AlphaData',~isnan(d_matrix));
set(gca,'XTickLabel',{},'XTick',1:length(BrainRegionLabel),'FontSize',2.5);
xtickangle(90)
set(gca, 'YTickLabel',{},'YTick',1:length(BrainRegionLabel),'FontSize',2.5);
set(gca,'TickLength',[0 0]);
FigName = fullfile(savefolder,strcat('d_matrix_SexDiff.png'));

[rows, column] = size(T_value);
Region_Index=180;
xline(Region_Index+0.5,'-k','LineWidth',0.9);
yline(Region_Index+0.5,'-k','LineWidth',0.9);

cb = colorbar;
cb.Label.String = 'Cohen''s d';
cb.Label.FontSize = 9;
cb.FontSize = 8;

colormap('jet');
caxis([-0.5 0.5]);

f = gcf;
set(f,'PaperUnits','centimeters','PaperPosition',[0 0 19 19]);
print(figure(1),FigName,'-dpng','-r600');




