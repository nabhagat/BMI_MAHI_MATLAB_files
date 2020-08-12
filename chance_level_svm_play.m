% Design classifier for Go vs No-Go classification of Movement Intention
% By Nikunj Bhagat, Graduate Student, University of Houston
% - 3/12/2014

% Modified to estimate chance level 4/3/2014
clear;

get_chance_level = 1;

%% Global Variables
myColors = ['g','r','m','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% Subject Details
Subject_name = 'MS';
Sess_num = 1;
Cond_num = 1;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 80;
use_phase_rand = 0;

folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
%folder_path = ['F:\Nikunj_Data\InMotion_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
% folder_path = ['D:\Subject_' Subject_name '\LG_Session4\'];

% if use_phase_rand == 1
%     Subject_name_old = Subject_name;
%     Subject_name = [Subject_name '_pr'];
% end

load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_average.mat']);
if get_chance_level == 1
    load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance.mat']);
end

Fs_eeg = 10; % Required
downsamp_factor = Average.Fs_eeg/Fs_eeg;
[no_epochs,no_datapts,no_channels] = size(Average.move_epochs);
for k = 1:no_channels
    move_epochs_s(:,:,k) = downsample(Average.move_epochs(:,:,k)',downsamp_factor)';
    rest_epochs_s(:,:,k) = downsample(Average.rest_epochs(:,:,k)',downsamp_factor)';
end
[no_epochs,no_datapts,no_channels] = size(move_epochs_s);
move_erp_time = downsample(Average.move_erp_time,downsamp_factor);
rest_erp_time = downsample(Average.rest_erp_time,downsamp_factor);

apply_epoch_standardization = 0; % Not possible in real-time
%1. Features to use
use_mahalanobis = 1;

%2. Training Classifier
move_window = [-0.7 -0.1];
rest_window = [-2.0 -1.4];
%rest_window = move_window;

%3. Classifier Channels
%classchannels = Average.RP_chans;
classchannels = Performance.classchannels;

%4. SVM scaling variables
apply_scaling = 1;
maxval = 1;
minval = 0;

%5. Classifier parameters
kernel_type = 2; % 2 - RBF
C = [10 100 1000];
gamma = [0.2 0.5 0.8 1];


if apply_epoch_standardization == 1
   for k = 1:no_channels
        move_epochs_s(:,:,k) = zscore(move_epochs_s(:,:,k),0,2);
        rest_epochs_s(:,:,k) = zscore(rest_epochs_s(:,:,k),0,2);
   end
end


%% Extract feature vectors from move and rest epochs
% Time Interval for training classifier
mlim1 = round(abs( move_erp_time(1)-(move_window(1)))*Fs_eeg+1);
mlim2 = round(abs( move_erp_time(1)-(move_window(2)))*Fs_eeg+1);
rlim1 = round(abs( move_erp_time(1)-(rest_window(1)))*Fs_eeg+1);
rlim2 = round(abs( move_erp_time(1)-(rest_window(2)))*Fs_eeg+1);
num_feature_per_chan = mlim2 - mlim1 + 1;

data_set = [];
%peaks = zeros(2*no_epochs,length(classchannels));
% AUC = zeros(2*no_epochs,length(classchannels));

for chan_ind = 1:length(classchannels)
    move_chn = move_epochs_s(:,:,classchannels(chan_ind));
    rest_chn = rest_epochs_s(:,:,classchannels(chan_ind));
    
    feature_space = [move_chn(:,mlim1:mlim2); rest_chn(:,rlim1:rlim2)];
    %peaks(:,chan_ind) = min(feature_space,[],2);
    %AUC(:,chan_ind) = [ trapz(move_erp_time(mlim1:mlim2),move_chn(:,mlim1:mlim2),2); ...
    %                    trapz(rest_erp_time(rlim1:rlim2),rest_chn(:,rlim1:rlim2),2)];
    data_set = [data_set feature_space];
end

data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

move_ch_avg = mean(move_epochs_s(:,:,classchannels),3);
rest_ch_avg = mean(rest_epochs_s(:,:,classchannels),3);

%1. Slope
data_set1 = [(move_ch_avg(:,mlim2) - move_ch_avg(:,mlim1))/(move_erp_time(mlim2) - move_erp_time(mlim1));
            (rest_ch_avg(:,rlim2) - rest_ch_avg(:,rlim1))/(rest_erp_time(rlim2) - rest_erp_time(rlim1))];

%2. Negative Peak 
data_set1 = [data_set1 [min(move_ch_avg(:,mlim1:mlim2),[],2); min(rest_ch_avg(:,rlim1:rlim2),[],2)]];

%3. Area under curve
data_set1 = [data_set1 [trapz(move_erp_time(mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)')';
                        trapz(rest_erp_time(rlim1:rlim2),rest_ch_avg(:,rlim1:rlim2)')']];

% %4. Mean
% data_set1 = [data_set1 [mean(move_ch_avg(:,mlim1:mlim2),2);
%                         mean(rest_ch_avg(:,rlim1:rlim2),2)]];

%5. Amplitude over interval
%data_set = [move_ch_avg(:,mlim1:mlim2); rest_ch_avg(:,rlim1:rlim2)];

%6. Mahalanobis distance of each trial from average over trials
mahal_dist = zeros(2*no_epochs,1);
for d = 1:no_epochs
    mahal_dist(d) = sqrt(mahal(move_ch_avg(d,mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)));
    mahal_dist(d + no_epochs) = sqrt(mahal(rest_ch_avg(d,rlim1:rlim2),move_ch_avg(:,mlim1:mlim2)));
end
%figure; plot(mahal_dist(1:no_epochs),'ob'); hold on; plot(mahal_dist(no_epochs+1:2*no_epochs),'xr');
if use_mahalanobis == 1
    data_set1 = [data_set1 mahal_dist];
end

scatter_set = data_set1;
%% Data Visualization
% % % 1. Scatter Plot
% % figure; 
% % h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,data_set_labels,'filled');
% % xlabel('Slope','FontSize',14);
% % ylabel('Mahalanobis','FontSize',14);
% % zlabel('AUC','FontSize',14);
% % s = 0.2;
% % %Obtain the axes size (in axpos) in Points
% % currentunits = get(gca,'Units');
% % set(gca, 'Units', 'Points');
% % axpos = get(gca,'Position');
% % set(gca, 'Units', currentunits);
% % 
% % markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
% % 
% % set(h, 'SizeData', markerWidth^8)
% % axis([-10 10 2 8 -6 4]);
% % 
% % %title('features 1 2 3');
% % % figure; scatter3(data_set1(:,2),data_set1(:,3),data_set1(:,4),10,data_set_labels);
% % % title('features 2 3 4');
% % % figure; scatter(data_set1(:,1),data_set1(:,2),10,data_set_labels);
% % 
% % %2. Raw Amplitudes
% % figure; 
% % for ntrial = 1:size(data_set,1)
% %     hold on;
% %     if data_set_labels(ntrial) == 1
% %         %plot(move_erp_time(find(move_erp_time == move_window(1)):find(move_erp_time == move_window(2))),...
% %          plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan));
% %     else
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan),'r')
% %     end
% %     grid on;
% %     set(gca,'YDir','reverse')
% %     axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
% %     hold off;
% % end

chance_accur = [];
for oloop = 1:500
    
%% Classifier Training and Cross-validation

% Empirical estimate of chance level.
% Ramdom Permutation data set labels
chance_set_labels = 2*ones(2*no_epochs,1);
chance_set_labels(randperm(2*no_epochs,no_epochs),1) = 1;


CVO = cvpartition(no_epochs,'leaveout');

% % lda_sensitivity = zeros(1,CVO.NumTestSets);
% % lda_specificity = zeros(1,CVO.NumTestSets);
% % lda_accur = zeros(1,CVO.NumTestSets);
% % % TPR = zeros(1,CVO.NumTestSets);
% % % FPR = zeros(1,CVO.NumTestSets);
% % 
% % for cv_index = 1:CVO.NumTestSets
% %     trIdx = CVO.training(cv_index);
% %     teIdx = CVO.test(cv_index);
% %     [test_labels, Error, Posterior, LogP, OutputCoefficients] = ...
% %         classify(data_set1([teIdx;teIdx],:),data_set1([trIdx;trIdx],:),data_set_labels([trIdx;trIdx],1), 'linear');
% % 
% %     CM = confusionmat(data_set_labels([teIdx;teIdx],1),test_labels);
% %     lda_sensitivity(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
% %     lda_specificity(cv_index) = CM(2,2)/(CM(2,2)+CM(2,1));
% % %     TPR(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
% % %     FPR(cv_index) = CM(2,1)/(CM(2,2)+CM(2,1));
% %     lda_accur(cv_index) = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2))*100;   
% % end

%lda_avg_accuracy = mean(accur)

%% SVM Classifier
% 1. Scale data set between minval to maxval
if  apply_scaling == 1
    [nr,nc] = size(data_set1);
    scale_range = zeros(2,nc);
    for k = 1:nc
        attribute = data_set1(:,k);
        scale_range(:,k) = [min(attribute); max(attribute)];
        attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval - minval) + minval;
        data_set1(:,k) = attribute;
    end
end    

% 2. Create sparse matrix
features_sparse = sparse(data_set1);           

% 3. Hyperparameter Optimization (C,gamma) via Grid Search
% %CV_grid = zeros(3,length(C)*length(gamma));
% CV_grid = [];
% for i = 1:length(C)
%     for j = 1:length(gamma)
%         CV_grid = [CV_grid [C(i) ; gamma(j); svmtrain(data_set_labels, features_sparse, ['-t ' num2str(kernel_type)...
%             ' -c ' num2str(C(i)) ' -g ' num2str(gamma(j)) ' -v 10'])]]; % C-SVC 
%     end
% end
% [hyper_m,hyper_i] = max(CV_grid(3,:)); 
% C_opt = CV_grid(1,hyper_i);
% gamma_opt = CV_grid(2,hyper_i);
C_opt = Performance.C_opt;
gamma_opt = Performance.gamma_opt;

% 4. Train SVM model and cross-validate
svm_sensitivity = zeros(1,CVO.NumTestSets);
svm_specificity = zeros(1,CVO.NumTestSets);
svm_accur = zeros(1,CVO.NumTestSets);
% TPR = zeros(1,CVO.NumTestSets);
% FPR = zeros(1,CVO.NumTestSets);

for cv_index = 1:CVO.NumTestSets
    trIdx = CVO.training(cv_index);
    teIdx = CVO.test(cv_index);
    
%     if use_phase_rand == 1
%         load([folder_path Subject_name_old '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_svmmodel.mat']);
%     else    
%         svm_model = svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
%                 ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt) ' -g ' num2str(gamma_opt) ' -b 1']);
            svm_model = svmtrain(chance_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt) ' -g ' num2str(gamma_opt) ' -b 1']);
%     end
            
%    [test_labels, accuracy, prob_estimates] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), svm_model, '-b 1');
    [test_labels, accuracy, prob_estimates] = svmpredict(chance_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), svm_model, '-b 1');
   
    %CM = confusionmat(data_set_labels([teIdx;teIdx],1),test_labels);
    CM = confusionmat(chance_set_labels([teIdx;teIdx],1),test_labels);
%    svm_sensitivity(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
%    svm_specificity(cv_index) = CM(2,2)/(CM(2,2)+CM(2,1));
%     TPR(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
%     FPR(cv_index) = CM(2,1)/(CM(2,2)+CM(2,1));

    if size(CM,1) < 2
        CM = [CM 0; 0 0];
    end
    svm_accur(cv_index) = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2))*100;   
end
  chance_accur(oloop,:) = svm_accur;

% SPECTF = csvread('SPECTF.train');
% labels = SPECTF(:, 1); % labels from the 1st column
% features = SPECTF(:, 2:end);
% features_sparse = sparse(features); % features must be in a sparse matrix
% libsvmwrite('SPECTFlibsvm.train', labels, features_sparse);
% 
% [heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');
% model = svmtrain(heart_scale_label, heart_scale_inst, '-t 2 -c 1 -g 0.07 -b 1');
% % [heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');
% [predict_label, accuracy, prob_estimates] = svmpredict(heart_scale_label, heart_scale_inst, model, '-b 1');
% disp(accuracy);
end
%% Save Results
filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_chance_dist.mat'];
save(filename2,'chance_accur');

%% Plot sensitivity & specificity
% group_names = {'SVM_Sen','SVM_Spe'};
% %subject_markers = {'+r','ok','*b','^r','xm','sy','dk','^r','vg','>b','<c','pm','hy','xk','+r'};
% figure;
% boxplot([svm_sensitivity' svm_specificity'] ,'labels',group_names,'widths',0.5);
%figure; boxplot(chance_accur);


