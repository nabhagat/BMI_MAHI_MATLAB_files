% Design classifier for Go vs No-Go classification of Movement Intention
% By Nikunj Bhagat, Graduate Student, University of Houston
% - 3/12/2014
clear;

%% Global Variables
myColors = ['g','r','m','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% Subject Details
Subject_name = 'MS';
Sess_num = 1;
Cond_num = 3;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
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
classchannels = Average.RP_chans;
%classchannels = [14,32,9,49];

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

%6. Use Hybrid Classification
hybrid_classifier = 1;

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
% % 1. Scatter Plot
% figure; 
% h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,data_set_labels,'filled');
% xlabel('Slope','FontSize',14);
% ylabel('Mahalanobis','FontSize',14);
% zlabel('AUC','FontSize',14);
% s = 0.2;
% %Obtain the axes size (in axpos) in Points
% currentunits = get(gca,'Units');
% set(gca, 'Units', 'Points');
% axpos = get(gca,'Position');
% set(gca, 'Units', currentunits);
% 
% markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
% 
% set(h, 'SizeData', markerWidth^8)
% axis([-10 10 2 8 -6 4]);

%title('features 1 2 3');
% figure; scatter3(data_set1(:,2),data_set1(:,3),data_set1(:,4),10,data_set_labels);
% title('features 2 3 4');
% figure; scatter(data_set1(:,1),data_set1(:,2),10,data_set_labels);

%% 2. Raw Amplitudes
figure; 
for ntrial = 1:size(data_set,1)
    hold on;
    if data_set_labels(ntrial) == 1
        %plot(move_erp_time(find(move_erp_time == move_window(1)):find(move_erp_time == move_window(2))),...
         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan));
         %plot(move_erp_time(mlim1:mlim2),data_set(ntrial,:),'.b');
    else
        plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan),'r')
        %plot(move_erp_time(mlim1:mlim2),data_set(ntrial,:),'.r')
    end
    grid on;
    set(gca,'YDir','reverse')
    %axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
    hold off;
end

%% Classifier Training and Cross-validation

CVO = cvpartition(no_epochs,'kfold',10);
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
% % 
% % %lda_avg_accuracy = mean(accur)

%% SVM Classifier for EEG
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
%CV_grid = zeros(3,length(C)*length(gamma));
CV_grid = [];
for i = 1:length(C)
    for j = 1:length(gamma)
        CV_grid = [CV_grid [C(i) ; gamma(j); svmtrain(data_set_labels, features_sparse, ['-t ' num2str(kernel_type)...
            ' -c ' num2str(C(i)) ' -g ' num2str(gamma(j)) ' -v 10'])]]; % C-SVC 
    end
end
[hyper_m,hyper_i] = max(CV_grid(3,:)); 
C_opt = CV_grid(1,hyper_i);
gamma_opt = CV_grid(2,hyper_i);

% 4. Train SVM model and cross-validate
% svm_sensitivity = zeros(1,CVO.NumTestSets);
% svm_specificity = zeros(1,CVO.NumTestSets);
% svm_accur = zeros(1,CVO.NumTestSets);
% TPR = zeros(1,CVO.NumTestSets);
% FPR = zeros(1,CVO.NumTestSets);

for cv_index = 1:CVO.NumTestSets
    trIdx = CVO.training(cv_index);
    teIdx = CVO.test(cv_index);
    
%     if use_phase_rand == 1
%         load([folder_path Subject_name_old '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_svmmodel.mat']);
%     else    
        eeg_svm_model{cv_index} = svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt) ' -g ' num2str(gamma_opt) ' -b 1']);
%     end
            
    [eeg_decision(:,cv_index), accuracy, eeg_prob_estimates(:,:,cv_index)] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), eeg_svm_model{cv_index}, '-b 1');
    
    eeg_CM(:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),eeg_decision(:,cv_index));
    
    %CM = confusionmat(data_set_labels([teIdx;teIdx],1),test_labels);
%     svm_sensitivity(cv_index) = eeg_CM(1,1,cv_index)/(eeg_CM(1,1,cv_index)+eeg_CM(1,2,cv_index));
%     svm_specificity(cv_index) = eeg_CM(2,2,cv_index)/(eeg_CM(2,2,cv_index)+eeg_CM(2,1,cv_index));
% %     TPR(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
% %     FPR(cv_index) = CM(2,1)/(CM(2,2)+CM(2,1));
%     svm_accur(cv_index) = (eeg_CM(1,1)+eeg_CM(2,2))/(eeg_CM(1,1)+eeg_CM(1,2)+eeg_CM(2,1)+eeg_CM(2,2))*100;   
    
    eeg_sensitivity(cv_index) = eeg_CM(1,1,cv_index)/(eeg_CM(1,1,cv_index)+eeg_CM(1,2,cv_index));
    eeg_specificity(cv_index) = eeg_CM(2,2,cv_index)/(eeg_CM(2,2,cv_index)+eeg_CM(2,1,cv_index));
    eeg_accur(cv_index) = (eeg_CM(1,1,cv_index)+eeg_CM(2,2,cv_index))/(eeg_CM(1,1,cv_index)+eeg_CM(1,2,cv_index)+eeg_CM(2,1,cv_index)+eeg_CM(2,2,cv_index))*100;   
end

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

%% EMG based Intention Classifier
% % % load('MS_ses1_cond1_block80_emg_move_epochs.mat');
% % % load('MS_ses1_cond1_block80_emg_rest_epochs.mat');
% % 
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_emg_epochs.mat']);

Fs_emg = emg_epochs.Fs_emg;
emg_move_epochs = emg_epochs.emg_move_epochs;

[no_epochs,no_datapts,no_channels] = size(emg_move_epochs);
emg_erp_time = emg_epochs.emg_erp_time; %round((-2.5:1/Fs_eeg:3.1)*100)./100;        % Get range of signal here ?? 4/12/14

figure; 
for i = 31:60
    subplot(2,1,1); hold on; grid on;
    plot(emg_erp_time,emg_move_epochs(i,:,1),'b');
    subplot(2,1,2); hold on; grid on;
    plot(emg_erp_time,emg_move_epochs(i,:,2),'r');
end
hold off;

bicep_thr = 10;
tricep_thr = 8;

move_window = [-0.7 -0.1];
rest_window = [-2.0 -1.4];
mlim1 = round(abs( emg_erp_time(1)-(move_window(1)))*Fs_emg+1);
mlim2 = round(abs( emg_erp_time(1)-(move_window(2)))*Fs_emg+1);
rlim1 = round(abs( emg_erp_time(1)-(rest_window(1)))*Fs_emg+1);
rlim2 = round(abs( emg_erp_time(1)-(rest_window(2)))*Fs_emg+1);

    
bicep_set = [emg_move_epochs(:,mlim1:mlim2,1);emg_move_epochs(:,rlim1:rlim2,1)];
tricep_set = [emg_move_epochs(:,mlim1:mlim2,2);emg_move_epochs(:,rlim1:rlim2,2)];

features_sparse = [];
emg_data_set = [max(bicep_set,[],2) max(tricep_set,[],2)];
emg_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];
emg_decision = [];

% for k = 1:2*no_epochs  
%     if find(bicep_set(k,:) >= bicep_thr) 
%         emg_decision(k) = 1;
%     elseif find(tricep_set(k,:) >= tricep_thr)
%         emg_decision(k) = 1;
%     else
%         emg_decision(k) = 2;
%     end
% end
   
% emg_CM = confusionmat(emg_labels,emg_decision);
% emg_sens = emg_CM(1,1)/(emg_CM(1,1)+emg_CM(1,2))
% emg_spec = emg_CM(2,2)/(emg_CM(2,2)+emg_CM(2,1))
% emg_accur = (emg_CM(1,1)+emg_CM(2,2))/(emg_CM(1,1)+emg_CM(1,2)+emg_CM(2,1)+emg_CM(2,2))*100 
% 
% filename11 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_emg_decision.mat'];
%              save(filename11,'emg_decision');

%------------------------------ SVM Classifier for EMG
% 1. Scale data set between minval to maxval
if  apply_scaling == 1
    [nr,nc] = size(emg_data_set);
    scale_range = zeros(2,nc);
    for k = 1:nc
        attribute = emg_data_set(:,k);
        scale_range(:,k) = [min(attribute); max(attribute)];
        attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval - minval) + minval;
        emg_data_set(:,k) = attribute;
    end
end    

% 2. Create sparse matrix
emg_features_sparse = sparse(emg_data_set);           

% 3. Hyperparameter Optimization (C,gamma) via Grid Search
%CV_grid = zeros(3,length(C)*length(gamma));
CV_grid = [];
for i = 1:length(C)
    for j = 1:length(gamma)
        CV_grid = [CV_grid [C(i) ; gamma(j); svmtrain(emg_labels, emg_features_sparse, ['-t ' num2str(kernel_type)...
            ' -c ' num2str(C(i)) ' -g ' num2str(gamma(j)) ' -v 10'])]]; % C-SVC 
    end
end
[hyper_m,hyper_i] = max(CV_grid(3,:)); 
emg_C_opt = CV_grid(1,hyper_i);
emg_gamma_opt = CV_grid(2,hyper_i);

% 4. Train SVM model and cross-validate
emg_sensitivity = zeros(1,CVO.NumTestSets);
emg_specificity = zeros(1,CVO.NumTestSets);
emg_accur = zeros(1,CVO.NumTestSets);
% TPR = zeros(1,CVO.NumTestSets);
% FPR = zeros(1,CVO.NumTestSets);

for cv_index = 1:CVO.NumTestSets
    trIdx = CVO.training(cv_index);
    teIdx = CVO.test(cv_index);
    
%     if use_phase_rand == 1
%         load([folder_path Subject_name_old '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_svmmodel.mat']);
%     else    
        emg_svm_model{cv_index} = svmtrain(emg_labels([trIdx;trIdx],1), emg_features_sparse([trIdx;trIdx],:),...
                ['-t ' num2str(kernel_type) ' -c ' num2str(emg_C_opt) ' -g ' num2str(emg_gamma_opt) ' -b 1']);
%     end
            
    [emg_decision(:,cv_index), accuracy, emg_prob_estimates(:,:,cv_index)] = svmpredict(emg_labels([teIdx;teIdx],1), emg_features_sparse([teIdx;teIdx],:), emg_svm_model{cv_index}, '-b 1');
       
    emg_CM(:,:,cv_index) = confusionmat(emg_labels([teIdx;teIdx],1),emg_decision(:,cv_index));
     
    emg_sensitivity(cv_index) = emg_CM(1,1,cv_index)/(emg_CM(1,1,cv_index)+emg_CM(1,2,cv_index));
    emg_specificity(cv_index) = emg_CM(2,2,cv_index)/(emg_CM(2,2,cv_index)+emg_CM(2,1,cv_index));
    emg_accur(cv_index) = (emg_CM(1,1,cv_index)+emg_CM(2,2,cv_index))/(emg_CM(1,1,cv_index)+emg_CM(1,2,cv_index)+emg_CM(2,1,cv_index)+emg_CM(2,2,cv_index))*100;

end

%% Hybrid Classifier
% Combine EEG and EMG decisions

for cv_index = 1:CVO.NumTestSets
    teIdx = CVO.test(cv_index);
    
    eeg_wt = max(eeg_prob_estimates(:,:,cv_index),[],2);       % TA - 0.5,0.5,0
    emg_wt = max(emg_prob_estimates(:,:,cv_index),[],2);
    comb_de_thr = -0.04;
    
    eeg_de = eeg_decision(:,cv_index);
    emg_de = emg_decision(:,cv_index); 
    eeg_de(eeg_de == 2) = -1;
    emg_de(emg_de == 2) = -1;
    
    comb_de = eeg_wt.*eeg_de + emg_wt.*emg_de;
    comb_de(comb_de>=comb_de_thr) = 1;
    comb_de(comb_de<comb_de_thr) = 2;
    
    comb_decision(:,cv_index) = comb_de;
    comb_CM(:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),comb_decision(:,cv_index));
    comb_sensitivity(cv_index) = comb_CM(1,1,cv_index)/(comb_CM(1,1,cv_index)+comb_CM(1,2,cv_index));
    comb_specificity(cv_index) = comb_CM(2,2,cv_index)/(comb_CM(2,2,cv_index)+comb_CM(2,1,cv_index));
    comb_accur(cv_index) = (comb_CM(1,1,cv_index)+comb_CM(2,2,cv_index))/(comb_CM(1,1,cv_index)+comb_CM(1,2,cv_index)+comb_CM(2,1,cv_index)+comb_CM(2,2,cv_index))*100;
end

%% Plot sensitivity & specificity
figure; 
subplot(1,3,1)
group_names = {'Sens','Spec'};
boxplot([eeg_sensitivity' eeg_specificity'] ,'labels',group_names,'widths',0.5);
title('Only EEG');
v = axis;
axis([v(1) v(2) 0.4 1.1]);

subplot(1,3,2)
%group_names = {'Sens','Spec'};
boxplot([emg_sensitivity' emg_specificity'] ,'labels',group_names,'widths',0.5);
title('Only EMG');
v = axis;
axis([v(1) v(2) 0.4 1.1]);

subplot(1,3,3)
%group_names = {'Sens','Spec'};
boxplot([comb_sensitivity' comb_specificity'] ,'labels',group_names,'widths',0.5);
%title(['EEG(' num2str(eeg_wt) ') + EMG(' num2str(emg_wt) '), Thr = ' num2str(comb_de_thr)]);
title('EEG + EMG');
v = axis;
axis([v(1) v(2) 0.4 1.1]);

%% Save Results
Performance = [];
Performance.apply_epoch_standardization = apply_epoch_standardization;
Performance.use_mahalanobis = use_mahalanobis;
Performance.move_window = move_window;
Performance.rest_window = rest_window;
Performance.classchannels = classchannels;
Performance.eeg_data_set = data_set1;
Performance.emg_data_set = emg_data_set;
Performance.data_set_labels = data_set_labels;

Performance.C_opt = C_opt;
Performance.gamma_opt = gamma_opt;
Performance.emg_C_opt = emg_C_opt;
Performance.emg_gamma_opt = emg_gamma_opt;

Performance.CVO = CVO;
% Performance.lda_sensitivity = lda_sensitivity;
% Performance.lda_specificity = lda_specificity;
% Performance.lda_accur = lda_accur;
% Performance.svm_sensitivity = svm_sensitivity;
% Performance.svm_specificity = svm_specificity;
% Performance.svm_accur = svm_accur;

Performance.eeg_svm_model = eeg_svm_model;
Performance.emg_svm_model = emg_svm_model;

Performance.eeg_prob_estimates = eeg_prob_estimates;
Performance.emg_prob_estimates = emg_prob_estimates;
Performance.eeg_decision = eeg_decision;
Performance.emg_decision = emg_decision;
Performance.comb_decision = comb_decision;
Performance.comb_de_thr = comb_de_thr;

Performance.eeg_CM = eeg_CM;
Performance.emg_CM = emg_CM;
Performance.comb_CM = comb_CM;

Performance.eeg_sensitivity = eeg_sensitivity;
Performance.eeg_specificity = eeg_specificity;
Performance.eeg_accur = eeg_accur;

Performance.emg_sensitivity = emg_sensitivity;
Performance.emg_specificity = emg_specificity;
Performance.emg_accur = emg_accur;

Performance.comb_sensitivity = comb_sensitivity;
Performance.comb_specificity = comb_specificity;
Performance.comb_accur = comb_accur;

filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_new.mat'];
save(filename2,'Performance');   
            

