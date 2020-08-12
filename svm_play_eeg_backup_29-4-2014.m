% Design classifier for Go vs No-Go classification of Movement Intention
% By Nikunj Bhagat, Graduate Student, University of Houston
% - 3/12/2014
clear;

%% Global Variables
myColors = ['g','r','m','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% Subject Details
Subject_name = 'TA';
Sess_num = 1;
Cond_num = 1;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 80;

folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
%folder_path = ['F:\Nikunj_Data\InMotion_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
% folder_path = ['D:\Subject_' Subject_name '\LG_Session4\'];
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_average_move1.mat']);

%1. Use separate move and rest epochs? 0 - No, use move epochs for both; 1 - yes
use_separate_move_rest_epochs = 1; 

%2. Training Classifier
move_window = [-0.7 -0.1];      % This will not matter if I use smart features
if use_separate_move_rest_epochs == 1
    %rest_window = move_window;
    rest_window = [-1.1 -0.5];
else
    rest_window = [-2.0 -1.5];
end

Fs_eeg = 10; % Required
downsamp_factor = Average.Fs_eeg/Fs_eeg;
[no_epochs,no_datapts,no_channels] = size(Average.move_epochs);

for k = 1:no_channels
    move_epochs_s(:,:,k) = downsample(Average.move_epochs(:,:,k)',downsamp_factor)';
    if use_separate_move_rest_epochs == 1
        rest_epochs_s(:,:,k) = downsample(Average.rest_epochs(:,:,k)',downsamp_factor)';
    else 
        rest_epochs_s(:,:,k) = downsample(Average.move_epochs(:,:,k)',downsamp_factor)';
    end
end

[no_epochs,no_datapts,no_channels] = size(move_epochs_s);
move_erp_time = downsample(Average.move_erp_time(:),downsamp_factor);

if use_separate_move_rest_epochs == 1
    rest_erp_time = downsample(Average.rest_erp_time,downsamp_factor);
else 
    rest_erp_time = downsample(Average.move_erp_time,downsamp_factor);
end

%Standardize Epochs? 
apply_epoch_standardization = 0; % Not possible in real-time
if apply_epoch_standardization == 1
   for k = 1:no_channels
        move_epochs_s(:,:,k) = zscore(move_epochs_s(:,:,k),0,2);
        rest_epochs_s(:,:,k) = zscore(rest_epochs_s(:,:,k),0,2);
   end
end

%Features to use
use_mahalanobis = 1;

%3. Classifier Channels
classchannels = Average.RP_chans(1:4);
%classchannels = [14,49,32,10];

%4. SVM scaling variables
apply_scaling = 1;
maxval = 1;
minval = 0;

%5. Classifier parameters
kernel_type = 2; % 2 - RBF
C = [10 100 1000];
gamma = [0.2 0.5 0.8 1];

%6. Use Hybrid Classification
hybrid_classifier = 0;

%7. Use new features
use_new_features = 0;
    keep_bad_trials = 0;
    % Find peak within interval, Reject trials reject_trial_onwards 
    find_peak_interval = [-2.0 0.0];
    reject_trial_before = -1.5; % Seconds. reject trials for which negative peak is reached earlier than -1.5 sec
    window_length = (move_window(2) - move_window(1))*Fs_eeg; % 5*100 = 500 msec 

%8. Cross validation approach
use_shifting_window_CV = 1;

%9. Baseline Correction? 
use_baseline_correction = 0;
% move_epochs_s - mean_move_baseline


%% Extract feature vectors from move and rest epochs -- Time domain features; Not used for classification
% Time Interval
mlim1 = round(abs( move_erp_time(1)-(move_window(1)))*Fs_eeg+1);
mlim2 = round(abs( move_erp_time(1)-(move_window(2)))*Fs_eeg+1);
rlim1 = round(abs( rest_erp_time(1)-(rest_window(1)))*Fs_eeg+1);
rlim2 = round(abs( rest_erp_time(1)-(rest_window(2)))*Fs_eeg+1);
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


%% Fixed window features

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

%% --------------------------- Smart features, 4/21/14
%if use_new_features == 1
% Reinitialize
move_ch_avg = [];
move_ch_avg_time = [];
rest_ch_avg = [];
rest_ch_avg_time = [];


move_ch_avg_ini = mean(move_epochs_s(:,:,classchannels),3);
rest_ch_avg_ini = mean(rest_epochs_s(:,:,classchannels),3);

mt1 = find(move_erp_time == find_peak_interval(1));
mt2 = find(move_erp_time == find_peak_interval(2));
[min_avg(:,1),min_avg(:,2)] = min(move_ch_avg_ini(:,mt1:mt2),[],2); % value, indices
bad_move_trials = []; 
good_move_trials = [];

for nt = 1:size(move_ch_avg_ini,1)
    if (move_erp_time(move_ch_avg_ini(nt,:) == min_avg(nt,1)) <= reject_trial_before)
        %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26),'r'); hold on;
        bad_move_trials = [bad_move_trials; nt];
    else
        %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26)); hold on;
        good_move_trials = [good_move_trials; nt];
    end
end

figure;
subplot(2,1,1); hold on;
plot(move_erp_time,move_ch_avg_ini(good_move_trials,:),'b');
title([Subject_name ', Cond ' num2str(Cond_num) ', # Good trials = ' num2str(length(good_move_trials)) ' (' num2str(size(move_ch_avg_ini,1)) ')'],'FontSize',12);
hold on;
plot(move_erp_time,move_ch_avg_ini(bad_move_trials,:),'r');
set(gca,'YDir','reverse');
grid on;
axis([-3.5 0.5 -15 10]);

subplot(2,1,2); hold on;
plot(rest_erp_time,rest_ch_avg_ini(good_move_trials,:),'b');
plot(rest_erp_time,rest_ch_avg_ini(bad_move_trials,:),'r');
set(gca,'YDir','reverse');
grid on;
axis([-3.5 0.5 -15 10]);

if keep_bad_trials == 1
    good_move_trials = 1:size(move_epochs_s,1);
end

for i = 1:length(good_move_trials)
    move_window_end = find(move_ch_avg_ini(good_move_trials(i),:) == min_avg(good_move_trials(i),1)); % upto Peak value
    move_window_start = move_window_end - window_length; 
    rest_window_start = find(rest_erp_time == rest_window(1));
    rest_window_end = rest_window_start + window_length;
    
    move_ch_avg(i,:) = move_ch_avg_ini(good_move_trials(i),move_window_start:move_window_end);
    move_ch_avg_time(i,:) = move_erp_time(move_window_start:move_window_end);
    subplot(2,1,1); hold on;
    plot(move_ch_avg_time(i,:),move_ch_avg(i,:),'k','LineWidth',2); hold on;  
    
    rest_ch_avg(i,:) = rest_ch_avg_ini(good_move_trials(i),rest_window_start:rest_window_end); 
    rest_ch_avg_time(i,:) = rest_erp_time(rest_window_start:rest_window_end);
    subplot(2,1,2); hold on;
    plot(rest_ch_avg_time(i,:),rest_ch_avg(i,:),'r','LineWidth',2); hold on;
end

% Reinitialize
no_epochs = size(move_ch_avg,1);        
data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

% figure; hold on;
% for j = 1:size(move_ch_avg,1)
%     plot((-0.5:0.1:0),move_ch_avg(j,:),'k','LineWidth',2);
% end
% set(gca,'YDir','reverse');
% grid on;
% axis([-0.5 0 -15 10]);

%1. Slope
data_set2 = [(move_ch_avg(:,end) - move_ch_avg(:,1))./(move_ch_avg_time(:,end) - move_ch_avg_time(:,1));
            (rest_ch_avg(:,end) - rest_ch_avg(:,1))./(rest_ch_avg_time(:,end) - rest_ch_avg_time(:,1))];

%2. Negative Peak 
data_set2 = [data_set2 [min(move_ch_avg,[],2); min(rest_ch_avg,[],2)]];

%3. Area under curve
for ind1 = 1:size(move_ch_avg,1)
    AUC_move(ind1) = trapz(move_ch_avg_time(ind1,:),move_ch_avg(ind1,:));
    AUC_rest(ind1) = trapz(rest_ch_avg_time(ind1,:),rest_ch_avg(ind1,:));
end
data_set2 = [data_set2 [AUC_move';AUC_rest']];
                       
%6. Mahalanobis distance of each trial from average over trials
mahal_dist2 = zeros(2*no_epochs,1);
for d = 1:no_epochs
    mahal_dist2(d) = sqrt(mahal(move_ch_avg(d,:),move_ch_avg(:,:)));
    mahal_dist2(d + no_epochs) = sqrt(mahal(rest_ch_avg(d,:),move_ch_avg(:,:)));
end
%figure; plot(mahal_dist2(1:no_epochs),'ob'); hold on; plot(mahal_dist2(no_epochs+1:2*no_epochs),'xr');
if use_mahalanobis == 1
    data_set2 = [data_set2 mahal_dist2];
end

%end

% if use_new_features == 1
%     classifier_dataset = data_set2;
% else
%     classifier_dataset = data_set1;
% end

%% Data Visualization

figure;

    scatter_set = data_set1;
    no_epochs = round(size(scatter_set,1)/2);
    subplot(1,2,1);
    % 1. Scatter Plot - data_Set1
    scolor =  [repmat([0 0 1],no_epochs,1);repmat([0.5 0.5 0.5],no_epochs,1)]; 
    h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,scolor,'filled');
    xlabel('Slope','FontSize',14);
    ylabel('Mahalanobis','FontSize',14);
    zlabel('AUC','FontSize',14);
    title('Fixed Interval Features','FontSize',14);
    s = 0.4;
    %Obtain the axes size (in axpos) in Points
    currentunits = get(gca,'Units');
    set(gca, 'Units', 'Points');
    axpos = get(gca,'Position');
    set(gca, 'Units', currentunits);
    markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
    set(h, 'SizeData', markerWidth^16)
    
    scatter_set = data_set2;
    no_epochs = round(size(scatter_set,1)/2);
    subplot(1,2,2);
    % 1. Scatter Plot - data_Set1
    scolor =  [repmat([0 0 1],no_epochs,1);repmat([0.5 0.5 0.5],no_epochs,1)]; 
    h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,scolor,'filled');
    xlabel('Slope','FontSize',14);
    ylabel('Mahalanobis','FontSize',14);
    zlabel('AUC','FontSize',14);
    title('Variable Interval Features','FontSize',14);
    s = 0.3;
    %Obtain the axes size (in axpos) in Points
    currentunits = get(gca,'Units');
    set(gca, 'Units', 'Points');
    axpos = get(gca,'Position');
    set(gca, 'Units', currentunits);
    markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
    set(h, 'SizeData', markerWidth^16)
    
    %axis([-10 10 2 8 -6 4]);
    %export_fig feature_space '-png' '-transparent';

    %subplot(2,2,2); 
    %scatter3(scatter_set(:,2),scatter_set(:,3),scatter_set(:,4),10,data_set_labels);
    %title('features 2 3 4');

% figure; scatter(classifier_dataset(:,1),classifier_dataset(:,2),10,data_set_labels);

% 2. Raw Amplitudes
% figure; 
% for ntrial = 1:size(data_set,1)
%     hold on;
%     if data_set_labels(ntrial) == 1
%         %plot(move_erp_time(find(move_erp_time == move_window(1)):find(move_erp_time == move_window(2))),...
%         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan));
%          %plot(move_erp_time(mlim1:mlim2),data_set(ntrial,:),'.b');
%     else
%         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan),'r')
%         %plot(move_erp_time(mlim1:mlim2),data_set(ntrial,:),'.r')
%     end
%     grid on;
%     set(gca,'YDir','reverse')
%     %axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
%     hold off;
% end


%% Classifier Training and Cross-validation
% % no_epochs = size(move_ch_avg,1);
% % CVO = cvpartition(no_epochs,'kfold',10);
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
% %         classify(classifier_dataset([teIdx;teIdx],:),classifier_dataset([trIdx;trIdx],:),data_set_labels([trIdx;trIdx],1), 'linear');
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

for use_feature = 2:2
    if use_feature == 1
        classifier_dataset = data_set1;
    elseif use_feature == 2
        classifier_dataset = data_set2;
    end
    
    no_epochs = round(size(classifier_dataset,1)/2);
    CVO = cvpartition(no_epochs,'kfold',10);
    data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];
    
    % 1. Scale data set between minval to maxval
    if  apply_scaling == 1
        [nr,nc] = size(classifier_dataset);
        scale_range = zeros(2,nc);
        for k = 1:nc
            attribute = classifier_dataset(:,k);
            scale_range(:,k) = [min(attribute); max(attribute)];
            attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval - minval) + minval;
            classifier_dataset(:,k) = attribute;
        end
    end    

    % 2. Create sparse matrix
    features_sparse = sparse(classifier_dataset);           

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
    C_opt(use_feature) = CV_grid(1,hyper_i);
    gamma_opt(use_feature) = CV_grid(2,hyper_i);

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
            eeg_svm_model{use_feature,cv_index} = svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                    ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt(use_feature)) ' -g ' num2str(gamma_opt(use_feature)) ' -b 1']);
    %     end
        t1 = -2.5;
        t2 = 0.0;
        if use_shifting_window_CV == 1
            % Sliding window validation 
            test_trials = [move_ch_avg_ini(find(teIdx==1),find(move_erp_time == t1):find(move_erp_time == t2));
                           rest_ch_avg_ini(find(teIdx==1),find(rest_erp_time == t1):find(rest_erp_time == t2))];                      
            test_trials_time = repmat(t1:1/Fs_eeg:t2,size(test_trials,1),1);
            test_trials_labels = data_set_labels([teIdx;teIdx],1);
            
            test_trial_decision = [];
            for test_trial_no = 1:size(test_trials,1)
                for win_start = 1:(size(test_trials,2) - window_length)
                        test_trial_signal_window(win_start,:) = test_trials(test_trial_no,win_start:(window_length + win_start));
                        test_trial_signal_time(win_start,:) = test_trials_time(test_trial_no,win_start:(window_length + win_start));
                end
                % Calculate features for each signal window
                test_trial_data_set = [];
                %1. Slope
                test_trial_data_set = (test_trial_signal_window(:,end) - test_trial_signal_window(:,1))./(test_trial_signal_time(:,end) - test_trial_signal_time(:,1));                         

                %2. Negative Peak 
                test_trial_data_set = [test_trial_data_set min(test_trial_signal_window,[],2)];

                %3. Area under curve
                for ind1 = 1:size(test_trial_signal_window,1)
                    AUC_tt(ind1) = trapz(test_trial_signal_time(ind1,:),test_trial_signal_window(ind1,:));
                end
                test_trial_data_set = [test_trial_data_set AUC_tt'];

                %6. Mahalanobis distance of each trial from average over trials
                test_mahal_dist = zeros(size(test_trial_signal_window,1),1);
                for d = 1:size(test_trial_signal_window,1)
                    test_mahal_dist(d) = sqrt(mahal(test_trial_signal_window(d,:),move_ch_avg(trIdx,:)));
                end
                %figure; plot(mahal_dist2(1:no_epochs),'ob'); hold on; plot(mahal_dist2(no_epochs+1:2*no_epochs),'xr');
                if use_mahalanobis == 1
                    test_trial_data_set = [test_trial_data_set test_mahal_dist];
                end
                
                % Apply SVM classifier
                % T1. Scale data set between minval to maxval
                if  apply_scaling == 1
                    [nr,nc] = size(test_trial_data_set);
                    scale_range = zeros(2,nc);
                    for k = 1:nc
                        attribute = test_trial_data_set(:,k);
                        scale_range(:,k) = [min(attribute); max(attribute)];
                        attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval - minval) + minval;
                        test_trial_data_set(:,k) = attribute;
                    end
                end    
                % T2. Create sparse matrix
                 test_trial_data_set = sparse(test_trial_data_set);    
                % T3. SVM prediction
                 internal_decision = svmpredict(repmat(test_trials_labels(test_trial_no),size(test_trial_data_set,1),1), test_trial_data_set, eeg_svm_model{use_feature,cv_index}, '-b 1');
                 
                 %-----------Find consecutive 1's >= Threshold
                 Decision_Thr = 4;                
                 test_trial_decision(test_trial_no,:) = [0 0];
                 mycounter = 0;
                 for i = 2:length(internal_decision)
                     if (internal_decision(i)== 1) && (internal_decision(i-1)==1)
                         mycounter = mycounter+1;
                     else
                         mycounter = 0;
                     end
                     if mycounter >= Decision_Thr
                        test_trial_decision(test_trial_no,:) = [1 test_trial_signal_time(i,end)];
                        break;
                     end
                 end
            end
            test_trial_decision(test_trial_decision(:,1) == 0,1) = 2;
            eeg_decision{use_feature,cv_index} = test_trial_decision(:,1);
            eeg_prob_estimates{use_feature,cv_index} = test_trial_decision(:,2);
        else
            % Regular validation
            [eeg_decision{use_feature,cv_index}, accuracy, eeg_prob_estimates{use_feature,cv_index}] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), eeg_svm_model{use_feature,cv_index}, '-b 1');
        end

        eeg_CM(use_feature,:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),eeg_decision{use_feature,cv_index});

        %CM = confusionmat(data_set_labels([teIdx;teIdx],1),test_labels);
    %     svm_sensitivity(cv_index) = eeg_CM(1,1,cv_index)/(eeg_CM(1,1,cv_index)+eeg_CM(1,2,cv_index));
    %     svm_specificity(cv_index) = eeg_CM(2,2,cv_index)/(eeg_CM(2,2,cv_index)+eeg_CM(2,1,cv_index));
    % %     TPR(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
    % %     FPR(cv_index) = CM(2,1)/(CM(2,2)+CM(2,1));
    %     svm_accur(cv_index) = (eeg_CM(1,1)+eeg_CM(2,2))/(eeg_CM(1,1)+eeg_CM(1,2)+eeg_CM(2,1)+eeg_CM(2,2))*100;   

        eeg_sensitivity(use_feature,cv_index) = eeg_CM(use_feature,1,1,cv_index)/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index));
        eeg_specificity(use_feature,cv_index) = eeg_CM(use_feature,2,2,cv_index)/(eeg_CM(use_feature,2,2,cv_index)+eeg_CM(use_feature,2,1,cv_index));
        eeg_accur(use_feature,cv_index) = (eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index)+eeg_CM(use_feature,2,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))*100;   
    end
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
% % % % % load('MS_ses1_cond1_block80_emg_move_epochs.mat');
% % % % % load('MS_ses1_cond1_block80_emg_rest_epochs.mat');
% % % % 
% % load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_emg_epochs.mat']);
% % 
% % Fs_emg = emg_epochs.Fs_emg;
% % emg_move_epochs = emg_epochs.emg_move_epochs;
% % 
% % [no_epochs,no_datapts,no_channels] = size(emg_move_epochs);
% % emg_erp_time = emg_epochs.emg_erp_time; %round((-2.5:1/Fs_eeg:3.1)*100)./100;        % Get range of signal here ?? 4/12/14
% % 
% % figure; 
% % for i = 31:60
% %     subplot(2,1,1); hold on; grid on;
% %     plot(emg_erp_time,emg_move_epochs(i,:,1),'b');
% %     subplot(2,1,2); hold on; grid on;
% %     plot(emg_erp_time,emg_move_epochs(i,:,2),'r');
% % end
% % hold off;
% % 
% % bicep_thr = 10;
% % tricep_thr = 8;
% % 
% % move_window = [-0.7 -0.1];
% % rest_window = [-2.0 -1.4];
% % mlim1 = round(abs( emg_erp_time(1)-(move_window(1)))*Fs_emg+1);
% % mlim2 = round(abs( emg_erp_time(1)-(move_window(2)))*Fs_emg+1);
% % rlim1 = round(abs( emg_erp_time(1)-(rest_window(1)))*Fs_emg+1);
% % rlim2 = round(abs( emg_erp_time(1)-(rest_window(2)))*Fs_emg+1);
% % 
% %     
% % bicep_set = [emg_move_epochs(:,mlim1:mlim2,1);emg_move_epochs(:,rlim1:rlim2,1)];
% % tricep_set = [emg_move_epochs(:,mlim1:mlim2,2);emg_move_epochs(:,rlim1:rlim2,2)];
% % 
% % features_sparse = [];
% % emg_data_set = [max(bicep_set,[],2) max(tricep_set,[],2)];
% % emg_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];
% % emg_decision = [];
% % 
% % % for k = 1:2*no_epochs  
% % %     if find(bicep_set(k,:) >= bicep_thr) 
% % %         emg_decision(k) = 1;
% % %     elseif find(tricep_set(k,:) >= tricep_thr)
% % %         emg_decision(k) = 1;
% % %     else
% % %         emg_decision(k) = 2;
% % %     end
% % % end
% %    
% % % emg_CM = confusionmat(emg_labels,emg_decision);
% % % emg_sens = emg_CM(1,1)/(emg_CM(1,1)+emg_CM(1,2))
% % % emg_spec = emg_CM(2,2)/(emg_CM(2,2)+emg_CM(2,1))
% % % emg_accur = (emg_CM(1,1)+emg_CM(2,2))/(emg_CM(1,1)+emg_CM(1,2)+emg_CM(2,1)+emg_CM(2,2))*100 
% % % 
% % % filename11 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_emg_decision.mat'];
% % %              save(filename11,'emg_decision');
% % 
% % %------------------------------ SVM Classifier for EMG
% % % 1. Scale data set between minval to maxval
% % if  apply_scaling == 1
% %     [nr,nc] = size(emg_data_set);
% %     scale_range = zeros(2,nc);
% %     for k = 1:nc
% %         attribute = emg_data_set(:,k);
% %         scale_range(:,k) = [min(attribute); max(attribute)];
% %         attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval - minval) + minval;
% %         emg_data_set(:,k) = attribute;
% %     end
% % end    
% % 
% % % 2. Create sparse matrix
% % emg_features_sparse = sparse(emg_data_set);           
% % 
% % % 3. Hyperparameter Optimization (C,gamma) via Grid Search
% % %CV_grid = zeros(3,length(C)*length(gamma));
% % CV_grid = [];
% % for i = 1:length(C)
% %     for j = 1:length(gamma)
% %         CV_grid = [CV_grid [C(i) ; gamma(j); svmtrain(emg_labels, emg_features_sparse, ['-t ' num2str(kernel_type)...
% %             ' -c ' num2str(C(i)) ' -g ' num2str(gamma(j)) ' -v 10'])]]; % C-SVC 
% %     end
% % end
% % [hyper_m,hyper_i] = max(CV_grid(3,:)); 
% % emg_C_opt = CV_grid(1,hyper_i);
% % emg_gamma_opt = CV_grid(2,hyper_i);
% % 
% % % 4. Train SVM model and cross-validate
% % emg_sensitivity = zeros(1,CVO.NumTestSets);
% % emg_specificity = zeros(1,CVO.NumTestSets);
% % emg_accur = zeros(1,CVO.NumTestSets);
% % % TPR = zeros(1,CVO.NumTestSets);
% % % FPR = zeros(1,CVO.NumTestSets);
% % 
% % for cv_index = 1:CVO.NumTestSets
% %     trIdx = CVO.training(cv_index);
% %     teIdx = CVO.test(cv_index);
% %     
% % %     if use_phase_rand == 1
% % %         load([folder_path Subject_name_old '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_svmmodel.mat']);
% % %     else    
% %         emg_svm_model{cv_index} = svmtrain(emg_labels([trIdx;trIdx],1), emg_features_sparse([trIdx;trIdx],:),...
% %                 ['-t ' num2str(kernel_type) ' -c ' num2str(emg_C_opt) ' -g ' num2str(emg_gamma_opt) ' -b 1']);
% % %     end
% %             
% %     [emg_decision(:,cv_index), accuracy, emg_prob_estimates(:,:,cv_index)] = svmpredict(emg_labels([teIdx;teIdx],1), emg_features_sparse([teIdx;teIdx],:), emg_svm_model{cv_index}, '-b 1');
% %        
% %     emg_CM(:,:,cv_index) = confusionmat(emg_labels([teIdx;teIdx],1),emg_decision(:,cv_index));
% %      
% %     emg_sensitivity(cv_index) = emg_CM(1,1,cv_index)/(emg_CM(1,1,cv_index)+emg_CM(1,2,cv_index));
% %     emg_specificity(cv_index) = emg_CM(2,2,cv_index)/(emg_CM(2,2,cv_index)+emg_CM(2,1,cv_index));
% %     emg_accur(cv_index) = (emg_CM(1,1,cv_index)+emg_CM(2,2,cv_index))/(emg_CM(1,1,cv_index)+emg_CM(1,2,cv_index)+emg_CM(2,1,cv_index)+emg_CM(2,2,cv_index))*100;
% % 
% % end

%% Hybrid Classifier
% Combine EEG and EMG decisions

% % for cv_index = 1:CVO.NumTestSets
% %     teIdx = CVO.test(cv_index);
% %     
% %     eeg_wt = max(eeg_prob_estimates(:,:,cv_index),[],2);       % TA - 0.5,0.5,0
% %     emg_wt = max(emg_prob_estimates(:,:,cv_index),[],2);
% %     comb_de_thr = -0.04;
% %     
% %     eeg_de = eeg_decision(:,cv_index);
% %     emg_de = emg_decision(:,cv_index); 
% %     eeg_de(eeg_de == 2) = -1;
% %     emg_de(emg_de == 2) = -1;
% %     
% %     comb_de = eeg_wt.*eeg_de + emg_wt.*emg_de;
% %     comb_de(comb_de>=comb_de_thr) = 1;
% %     comb_de(comb_de<comb_de_thr) = 2;
% %     
% %     comb_decision(:,cv_index) = comb_de;
% %     comb_CM(:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),comb_decision(:,cv_index));
% %     comb_sensitivity(cv_index) = comb_CM(1,1,cv_index)/(comb_CM(1,1,cv_index)+comb_CM(1,2,cv_index));
% %     comb_specificity(cv_index) = comb_CM(2,2,cv_index)/(comb_CM(2,2,cv_index)+comb_CM(2,1,cv_index));
% %     comb_accur(cv_index) = (comb_CM(1,1,cv_index)+comb_CM(2,2,cv_index))/(comb_CM(1,1,cv_index)+comb_CM(1,2,cv_index)+comb_CM(2,1,cv_index)+comb_CM(2,2,cv_index))*100;
% % end

%% Plot sensitivity & specificity
figure; 
subplot(1,2,1)
group_names = {'Sens','Spec','Accur'};
boxplot([eeg_sensitivity(1,:)' eeg_specificity(1,:)' (eeg_accur(1,:)'./100)] ,'labels',group_names,'widths',0.5);
title('Fixed Interval Features');
%v = axis;
%axis([v(1) v(2) 0.4 1.1]);

subplot(1,2,2)
group_names = {'Sens','Spec','Accur'};
boxplot([eeg_sensitivity(2,:)' eeg_specificity(2,:)' (eeg_accur(2,:)'./100)] ,'labels',group_names,'widths',0.5);
title('Variable Interval Features');
%v = axis;
%axis([v(1) v(2) 0.4 1.1]);

% % 
% % subplot(1,3,2)
% % %group_names = {'Sens','Spec'};
% % boxplot([emg_sensitivity' emg_specificity'] ,'labels',group_names,'widths',0.5);
% % title('Only EMG');
% % v = axis;
% % axis([v(1) v(2) 0.4 1.1]);
% % 
% % subplot(1,3,3)
% % %group_names = {'Sens','Spec'};
% % boxplot([comb_sensitivity' comb_specificity'] ,'labels',group_names,'widths',0.5);
% % %title(['EEG(' num2str(eeg_wt) ') + EMG(' num2str(emg_wt) '), Thr = ' num2str(comb_de_thr)]);
% % title('EEG + EMG');
% % v = axis;
% % axis([v(1) v(2) 0.4 1.1]);

%% Save Results
Performance = [];
% Fixed paramters
Performance.apply_epoch_standardization = apply_epoch_standardization;
Performance.use_mahalanobis = use_mahalanobis;

% Variable Parameters
Performance.use_separate_move_rest_epochs = use_separate_move_rest_epochs;
Performance.move_window = move_window;
Performance.rest_window = rest_window;
Performance.classchannels = classchannels;

Performance.use_new_features = use_new_features;
Performance.keep_bad_trials = keep_bad_trials;
Performance.find_peak_interval = find_peak_interval;
Performance.reject_trial_before = reject_trial_before;
Performance.window_length = window_length;

Performance.eeg_data_set = data_set1;       % Old Features
Performance.eeg_data_set = data_set2;       % New Features
% Performance.emg_data_set = emg_data_set;
Performance.data_set_labels = data_set_labels;

Performance.C_opt = C_opt;
Performance.gamma_opt = gamma_opt;
% Performance.emg_C_opt = emg_C_opt;
% Performance.emg_gamma_opt = emg_gamma_opt;

Performance.CVO = CVO;
% Performance.lda_sensitivity = lda_sensitivity;
% Performance.lda_specificity = lda_specificity;
% Performance.lda_accur = lda_accur;
% Performance.svm_sensitivity = svm_sensitivity;
% Performance.svm_specificity = svm_specificity;
% Performance.svm_accur = svm_accur;

Performance.eeg_svm_model = eeg_svm_model;
% Performance.emg_svm_model = emg_svm_model;

Performance.eeg_prob_estimates = eeg_prob_estimates;
% Performance.emg_prob_estimates = emg_prob_estimates;
Performance.eeg_decision = eeg_decision;
% Performance.emg_decision = emg_decision;
% Performance.comb_decision = comb_decision;
% Performance.comb_de_thr = comb_de_thr;

Performance.eeg_CM = eeg_CM;
% Performance.emg_CM = emg_CM;
% Performance.comb_CM = comb_CM;

Performance.eeg_sensitivity = eeg_sensitivity;
Performance.eeg_specificity = eeg_specificity;
Performance.eeg_accur = eeg_accur;

% Performance.emg_sensitivity = emg_sensitivity;
% Performance.emg_specificity = emg_specificity;
% Performance.emg_accur = emg_accur;
% 
% Performance.comb_sensitivity = comb_sensitivity;
% Performance.comb_specificity = comb_specificity;
% Performance.comb_accur = comb_accur;

% filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_new.mat'];
% save(filename2,'Performance');   
            

