% Train classifier (SVM) for Go (Move) vs No-Go(Rest) classification of Movement Intention
% By Nikunj Bhagat, Graduate Student, University of Houston
% Date Created - 3/12/2014

%% ***********Revisions 
% 6/05/14 - Changed filename from 'svm_play_eeg.m' to 'Mahi_train_SVM'   
% 6/06/14 - Always use Mahalanobis Distance,
%         - Commented computation of signal amplitude as features
% 6/07/14 - Added variables to be saved
% 6/10/14 - Change desired real-time frequency to 20 hz (50 ms)
% 6/28/14 - Saving Covariance matrix and mean vector for smart and conventional features 
%           to directly compute Mahalanobis distance
% 1/3/15 - Added switch statement to select loop_start:loop_end in FOR
%                  loops
%               - Added option to load previously computed Performance variable
% 2/15/15 - Added option to switch between training of SVM and Sparse Representation Classifier (SRC) 
% 7/27/15 - Tried channel combinations using mRMR algorithm - commented 8/19/15
% 8/20/15 - Automatically select window length with criteria: minimum window length, such that AUC greater than 95% of maximum AUC. 
clear;
%close all;
%% Global Variables
myColors = ['g','r','m','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% Subject Details - change9
Subject_name = 'S9017';
Sess_num = '2';
Cond_num = 1;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;

folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change10
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_average_causal.mat']);      % Always use causal for training classifier

% Flags to control the Classifier Training
%-. Use separate move and rest epochs? (Always yes). 0 - No; 1 - yes
use_separate_move_rest_epochs = 1; 

Fs_eeg = 20; % Desired sampling frequency in real-time
downsamp_factor = Average.Fs_eeg/Fs_eeg;
[no_epochs,no_datapts,no_channels] = size(Average.move_epochs); %#ok<*ASGLU>

%-. Apply Baseline Correction? 
    apply_baseline_correction = 0;  % Always 0 for newer algorithms and also for EMBS
    if apply_baseline_correction == 1
        for epoch_cnt = 1:no_epochs
            for channel_cnt = 1:no_channels
                move_epochs(epoch_cnt,:,channel_cnt) = Average.move_epochs(epoch_cnt,:,channel_cnt)...
                    - Average.move_mean_baseline(channel_cnt,epoch_cnt);
                rest_epochs(epoch_cnt,:,channel_cnt) = Average.rest_epochs(epoch_cnt,:,channel_cnt)...
                    - Average.rest_mean_baseline(channel_cnt,epoch_cnt);
            end
        end
    else
        move_epochs = Average.move_epochs;
        rest_epochs = Average.rest_epochs;
    end

% Downsample the epochs and epoch_times
    for k = 1:no_channels
        move_epochs_s(:,:,k) = downsample(move_epochs(:,:,k)',downsamp_factor)';
        if use_separate_move_rest_epochs == 1
            rest_epochs_s(:,:,k) = downsample(rest_epochs(:,:,k)',downsamp_factor)';
        else 
            rest_epochs_s(:,:,k) = downsample(move_epochs(:,:,k)',downsamp_factor)';
        end
    end
    [no_epochs,no_datapts,no_channels] = size(move_epochs_s);
    move_erp_time = downsample(Average.move_erp_time(:),downsamp_factor);
    if use_separate_move_rest_epochs == 1
        rest_erp_time = downsample(Average.rest_erp_time,downsamp_factor);
    else 
        rest_erp_time = downsample(Average.move_erp_time,downsamp_factor);
    end

%-. Standardize Epochs? - No longer used
    apply_epoch_standardization = 0; % Not possible in real-time
    if apply_epoch_standardization == 1
       for k = 1:no_channels
            move_epochs_s(:,:,k) = zscore(move_epochs_s(:,:,k),0,2);
            rest_epochs_s(:,:,k) = zscore(rest_epochs_s(:,:,k),0,2);
       end
    end

%0. Use previously trained models? Ex: CVO, smart_features, etc. 
use_previous_models = 0;    % change11
regular_or_chance_level_classifier = 1; % 0 - chance_level; 1 - Regular classifier design
use_conventional_features = 1;              % change12                              
use_smart_features = 1;

if use_previous_models == 1
    if regular_or_chance_level_classifier == 1
        prev_Performance = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) ...
            '_block' num2str(Block_num) '_performance_optimized_causal.mat']);  
    elseif regular_or_chance_level_classifier == 0
        if use_conventional_features == 1
            prev_Performance = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) ...
                '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);
            classification_error = 1 - mean(prev_Performance.All_eeg_accur{prev_Performance.conv_opt_wl_ind})/100;
        elseif use_smart_features == 1
            prev_Performance = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) ...
                '_block' num2str(Block_num) '_performance_optimized_causal.mat']);
            classification_error = 1 - mean(prev_Performance.eeg_accur)/100;
        end
    end
end
    
%1. Classifier Channels
use_channel_optimization = 1;       % change14
if use_previous_models == 1
    classchannels = prev_Performance.classchannels;
     % classchannels = Average.RP_chans; %test_change
     % classchannels = [32 13 48 14 49 19];
else
    %classchannels = Average.RP_chans(1:4); % Needs to be optimized
    classchannels = Average.RP_chans;           % Automatic channel selection
    % classchannels = [32 14 49 52 54];         % Manual channel selection
end

% Added for MRMR algorithm - 7/27/2015, 8/19/2015
% Average.RP_chans = [Average.RP_chans [24 25 26 57 58]];
% Average.RP_chans = [43, 9, 32, 10, 44, 13, 48, 14, 49, 15, 52, 19, ... 
%                53, 20, 54, 24, 57, 25, 58, 26, 4, 38, 5, 39, 6];

%2. Classifier parameters
    kernel_type = 2; % 2 - RBF
    C = [10 100 1000];
    gamma = [0.2 0.5 0.8 1];
    apply_scaling_for_SVM = 0;      % 1 - For EMBS paper, 0 for newer algorithms
    maxval_SVM_scaling = 1;
    minval_SVM_scaling = 0;

%-. Use Hybrid Classification
    use_svm_classifier = 1;
    use_src_classifier = 0;

%3. Use Smart/Conventional features
    keep_bad_trials = 0;   % Remove bad trials only from Smart Features
    % Find peak within interval, Reject trials reject_trial_onwards 
    find_peak_interval = [-2.0 0.5];            % ranges up to 0.5 sec after movement onset because of phase delay filtering
    reject_trial_before = -1.5; % Seconds. reject trials for which negative peak is reached earlier than -1.5 sec
    
    % Decide loop_start, loop_end values
    switch use_conventional_features
        case 1
            switch use_smart_features
                case 1
                    loop_start = 1; loop_end = 2; 
                case 0
                    loop_start = 1; loop_end = 1;
            end
        case 0
            switch use_smart_features
                case 1
                    loop_start = 2; loop_end = 2; 
                case 0
                    error('Error!! Atleast one of use_conventional_features/use_smart_features must be set');
            end    
    end
     
%4. Cross validation approach
    use_shifting_window_CV = 1;  % 0 - Conventional CrossValidation, 1 - Shifting Window CrossValidation 
    crossvalidation_trial_time = [-2.5 0.95]; % Time interval over which cross validation is done for each trial 
    consecutive_cnts_threshold = 3;                         % To be Optimized
    prob_est_threshold = 0.5;                               % Initial threshold. Automatically optimized using ROC curves for window length
    Max_Min_Thr = [];
    test_trials_time = [];
    test_trial_signal_time = [];

% Compute Spatial Average (Mean Filtering) of chosen classifier channels
    move_ch_avg = mean(move_epochs_s(:,:,classchannels),3);
    rest_ch_avg = mean(rest_epochs_s(:,:,classchannels),3);
    
if regular_or_chance_level_classifier == 1
       
%5. Window Length Optimization Loop Starts -----------------------------    
window_length_range = [10:20]; %round(0.6*Fs_eeg); %[1:17];     % change13
%window_length_range = 10; %test_change
optimized_channels = cell(length(window_length_range),1);

for wl = 1:length(window_length_range) 
% Initialize variables
    %data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)]; 
    [no_epochs,no_datapts,no_channels] = size(Average.move_epochs);
    Conventional_Features = [];
    Smart_Features = [];
    smart_Mu_move = []; smart_Cov_Mat = [];
    conv_Mu_move = []; conv_Cov_Mat = [];
    bad_move_trials = []; 
    good_move_trials = [];
    AUC_move = [];
    AUC_rest = [];
    
    window_length = window_length_range(wl);    %  = (window_length/20) seconds 
    %window_length = (move_window(2) - move_window(1))*Fs_eeg; % 5*100 = 500 msec 
    
%6. Move and Rest Windows
    move_window_stop_time = 0.50;       % Only required to be decided for conventional features. (Median EMG onset time)
    move_window_start_time = move_window_stop_time - window_length/Fs_eeg;
    move_window = [move_window_start_time move_window_stop_time];       
                                    
    if use_separate_move_rest_epochs == 1
        rest_window_stop_time = -0.5;
        rest_window_start_time = rest_window_stop_time - window_length/Fs_eeg;
        rest_window = [rest_window_start_time rest_window_stop_time];
        %rest_window = [-1.5 -0.9];  
    else
        % For EMBS paper
        %rest_window_start_time = -3.5; 
        %rest_window_stop_time = rest_window_start_time + window_length/Fs_eeg; 
        
        % For newer algorithms
        rest_window_stop_time = -2.0;
        rest_window_start_time = rest_window_stop_time - window_length/Fs_eeg;
        rest_window = [rest_window_start_time rest_window_stop_time];
        %rest_window = [-2.0 -1.4];      
    end

%11. Time Interval and windows
    mlim1 = round(abs( move_erp_time(1)-(move_window(1)))*Fs_eeg+1);
    mlim2 = round(abs( move_erp_time(1)-(move_window(2)))*Fs_eeg+1);
    rlim1 = round(abs( rest_erp_time(1)-(rest_window(1)))*Fs_eeg+1);
    rlim2 = round(abs( rest_erp_time(1)-(rest_window(2)))*Fs_eeg+1);    
%% Fixed window (Conventional) features
if use_conventional_features == 1   
%1. Slope
    Conventional_Features = ...
        [(move_ch_avg(:,mlim2) - move_ch_avg(:,mlim1))/(move_erp_time(mlim2) - move_erp_time(mlim1));
         (rest_ch_avg(:,rlim2) - rest_ch_avg(:,rlim1))/(rest_erp_time(rlim2) - rest_erp_time(rlim1))];

%2. Negative Peak 
    Conventional_Features = ...
        [Conventional_Features [min(move_ch_avg(:,mlim1:mlim2),[],2); min(rest_ch_avg(:,rlim1:rlim2),[],2)]];

%3. Area under curve
    Conventional_Features = ...
        [Conventional_Features [trapz(move_erp_time(mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)')';
                                trapz(rest_erp_time(rlim1:rlim2),rest_ch_avg(:,rlim1:rlim2)')']];

%4. Mean
%   data_set1 = [data_set1 [mean(move_ch_avg(:,mlim1:mlim2),2);
%                           mean(rest_ch_avg(:,rlim1:rlim2),2)]];

%5. Amplitude over interval
%   data_set = [move_ch_avg(:,mlim1:mlim2); rest_ch_avg(:,rlim1:rlim2)];

%6. Mahalanobis distance of each trial from average over trials
    mahal_dist = zeros(2*no_epochs,1);
    for d = 1:no_epochs
        mahal_dist(d) = sqrt(mahal(move_ch_avg(d,mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)));
        mahal_dist(d + no_epochs) = sqrt(mahal(rest_ch_avg(d,rlim1:rlim2),move_ch_avg(:,mlim1:mlim2)));
    end
    
    % Direct computation of Mahalanobis distance
    conv_mahal_dist = zeros(2*no_epochs,1);
    conv_Cov_Mat = cov(move_ch_avg(:,mlim1:mlim2));
    conv_Mu_move = mean(move_ch_avg(:,mlim1:mlim2),1);
    for d = 1:no_epochs
        x = move_ch_avg(d,mlim1:mlim2);
        conv_mahal_dist(d) = sqrt((x-conv_Mu_move)/(conv_Cov_Mat)*(x-conv_Mu_move)');
        y = rest_ch_avg(d,rlim1:rlim2);
        conv_mahal_dist(d + no_epochs) = sqrt((y-conv_Mu_move)/(conv_Cov_Mat)*(y-conv_Mu_move)');
    end
    
     
%7. Amplitude Range
%   amp_range = [(max(bmove_ch_avg,[],2) - min(bmove_ch_avg,[],2));
%                (max(brest_ch_avg,[],2) - min(brest_ch_avg,[],2))];

end
%% Variable window (Smart) features, 4/21/14
if use_smart_features == 1
% Reinitialize - Needs to be modified
smart_move_ch_avg = [];
move_ch_avg_time = [];
smart_rest_ch_avg = [];
rest_ch_avg_time = [];

% Comment block open ---- 8/19/2015
% % old_classchannels = classchannels;
% % % Different combinations of eeg channels  - 7/27/2015
% % % total_no_combs = 0;           % commented 8/19/2015
% % % for i = 4:length(Average.RP_chans)
% % %     total_no_combs = total_no_combs + nchoosek(length(Average.RP_chans),i);
% % % end
% % total_no_combs = 25; % added 8/19/2015
% % Channel_comb_matrix = zeros(total_no_combs,length(Average.RP_chans));
% % Mahalanobis_comb_matrix = zeros(total_no_combs,2*no_epochs);
% % Slope_comb_matrix = zeros(total_no_combs,2*no_epochs);
% % Area_comb_matrix = zeros(total_no_combs,2*no_epochs);
% % Peak_comb_matrix = zeros(total_no_combs,2*no_epochs);
% % 
% % first_elm = 1;
% % last_elm = 0;
% % % for no_chans = 4:length(Average.RP_chans) 
% % %     % generate combinations - commented 8/19/2015
% % %     chans_combn = nchoosek(Average.RP_chans,no_chans);
% % %     last_elm = first_elm + size(chans_combn,1) - 1;
% % %     Channel_comb_matrix(first_elm:last_elm,1:size(chans_combn,2)) = chans_combn;
% % %     first_elm = last_elm + 1; 
% % % end
% % Channel_comb_matrix(:,1) = Average.RP_chans';   % added 8/19/2015
% % 
% %     for comb = 1:size(Channel_comb_matrix,1)
% %          Smart_Features = [];
% %          smart_Mu_move = []; smart_Cov_Mat = [];
% %          conv_Mu_move = []; conv_Cov_Mat = [];
% %          bad_move_trials = []; 
% %          good_move_trials = [];
% %          
% %         classchannels = Channel_comb_matrix(comb,:);
% %         classchannels = classchannels(classchannels~=0);
% Comment block close ---- 8/19/2015

    mt1 = find(move_erp_time == find_peak_interval(1));
    mt2 = find(move_erp_time == find_peak_interval(2));
    if use_channel_optimization == 1
        [optimized_channels,rejected_channel_pool,prev_MI]  = ...
            optimized_channel_selection_backward(classchannels,move_epochs_s,rest_epochs_s,move_erp_time,rest_erp_time,...
                                                                                        find_peak_interval,reject_trial_before, keep_bad_trials, window_length,rest_window);        %test_change

        move_ch_avg_ini = mean(move_epochs_s(:,:,optimized_channels),3);
        rest_ch_avg_ini = mean(rest_epochs_s(:,:,optimized_channels),3);
    else
        move_ch_avg_ini = mean(move_epochs_s(:,:,classchannels),3);
        rest_ch_avg_ini = mean(rest_epochs_s(:,:,classchannels),3);
    end
        
    [min_avg(:,1),min_avg(:,2)] = min(move_ch_avg_ini(:,mt1:mt2),[],2); % value, indices

    for nt = 1:size(move_ch_avg_ini,1)
        if (move_erp_time(move_ch_avg_ini(nt,:) == min_avg(nt,1)) <= reject_trial_before) %|| (min_avg(nt,1) > -3)        
            %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26),'r'); hold on;
            bad_move_trials = [bad_move_trials; nt];
        else
            %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26)); hold on;
            good_move_trials = [good_move_trials; nt];
        end
    end

        if wl == length(window_length_range) 
            figure;
            subplot(2,1,1); hold on;
            plot(move_erp_time,move_ch_avg_ini(good_move_trials,:),'b');
            title([Subject_name ', Cond ' num2str(Cond_num) ', # Good trials = ' num2str(length(good_move_trials)) ' (' num2str(size(move_ch_avg_ini,1)) ')'],'FontSize',12);
            hold on;
            plot(move_erp_time,move_ch_avg_ini(bad_move_trials,:),'r');
            set(gca,'YDir','reverse');
            grid on;
            axis([-3.5 1 -15 10]);
        
            subplot(2,1,2); hold on;
            plot(rest_erp_time,rest_ch_avg_ini(good_move_trials,:),'b');
            plot(rest_erp_time,rest_ch_avg_ini(bad_move_trials,:),'r');
            set(gca,'YDir','reverse');
            grid on;
            axis([-3.5 1 -15 10]);
        end

        if keep_bad_trials == 1
            good_move_trials = 1:size(move_epochs_s,1);
            good_trials_move_ch_avg = move_ch_avg_ini(good_move_trials,:);
            good_trials_rest_ch_avg = rest_ch_avg_ini(good_move_trials,:);
        else
            % Else remove bad trials from Conventional Features, move_ch_avg and
            % rest_ch_avg
            % Commented on 6/18/14
            % Conventional_Features([bad_move_trials; (size(Conventional_Features,1)/2)+bad_move_trials],:) = [];
        %     good_trials_move_ch_avg = move_ch_avg(good_move_trials,:);
        %     good_trials_rest_ch_avg = rest_ch_avg(good_move_trials,:);
        end

        for i = 1:length(good_move_trials)
            move_window_end = find(move_ch_avg_ini(good_move_trials(i),:) == min_avg(good_move_trials(i),1)); % index of Peak(min) value
            move_window_start = move_window_end - window_length; 
            rest_window_start = find(rest_erp_time == rest_window(1));
            rest_window_end = rest_window_start + window_length;

            smart_move_ch_avg(i,:) = move_ch_avg_ini(good_move_trials(i),move_window_start:move_window_end);
            move_ch_avg_time(i,:) = move_erp_time(move_window_start:move_window_end);
            smart_rest_ch_avg(i,:) = rest_ch_avg_ini(good_move_trials(i),rest_window_start:rest_window_end); 
            rest_ch_avg_time(i,:) = rest_erp_time(rest_window_start:rest_window_end);

        if wl == length(window_length_range) 
           subplot(2,1,1); hold on;
           plot(move_ch_avg_time(i,:),smart_move_ch_avg(i,:),'k','LineWidth',2); hold on;   
           subplot(2,1,2); hold on;
           plot(rest_ch_avg_time(i,:),smart_rest_ch_avg(i,:),'r','LineWidth',2); hold on;
        end

        end

        % Reinitialize
        no_epochs = size(smart_move_ch_avg,1);        
        data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

        % figure; hold on;
        % for j = 1:size(move_ch_avg,1)
        %     plot((-0.5:0.1:0),move_ch_avg(j,:),'k','LineWidth',2);
        % end
        % set(gca,'YDir','reverse');
        % grid on;
        % axis([-0.5 0 -15 10]);

        %1. Slope
        Smart_Features = [(smart_move_ch_avg(:,end) - smart_move_ch_avg(:,1))./(move_ch_avg_time(:,end) - move_ch_avg_time(:,1));
                          (smart_rest_ch_avg(:,end) - smart_rest_ch_avg(:,1))./(rest_ch_avg_time(:,end) - rest_ch_avg_time(:,1))];

        %2. Negative Peak 
        Smart_Features = [Smart_Features [min(smart_move_ch_avg,[],2); min(smart_rest_ch_avg,[],2)]];

        %3. Area under curve
        for ind1 = 1:size(smart_move_ch_avg,1)
            AUC_move(ind1) = trapz(move_ch_avg_time(ind1,:),smart_move_ch_avg(ind1,:));
            AUC_rest(ind1) = trapz(rest_ch_avg_time(ind1,:),smart_rest_ch_avg(ind1,:));
        end
        Smart_Features = [Smart_Features [AUC_move';AUC_rest']];

        %6. Mahalanobis distance of each trial from average over trials
            % Use formula for computation of Mahalanobis distance
            mahal_dist2 = zeros(2*no_epochs,1);
            for d = 1:no_epochs
                mahal_dist2(d) = sqrt(mahal(smart_move_ch_avg(d,:),smart_move_ch_avg(:,:)));
                mahal_dist2(d + no_epochs) = sqrt(mahal(smart_rest_ch_avg(d,:),smart_move_ch_avg(:,:)));
            end

            % Direct computation of Mahalanobis distance
            smart_mahal_dist = zeros(2*no_epochs,1);
            smart_Cov_Mat = cov(smart_move_ch_avg);
            smart_Mu_move = mean(smart_move_ch_avg,1);
            for d = 1:no_epochs
                x = smart_move_ch_avg(d,:);
                smart_mahal_dist(d) = sqrt((x-smart_Mu_move)/(smart_Cov_Mat)*(x-smart_Mu_move)');
                y = smart_rest_ch_avg(d,:);
                smart_mahal_dist(d + no_epochs) = sqrt((y-smart_Mu_move)/(smart_Cov_Mat)*(y-smart_Mu_move)');
            end
            Smart_Features = [Smart_Features smart_mahal_dist];
            
            % Comment from here 7/27/2015 - commented on 8/19/2015
%             Mahalanobis_comb_matrix(comb,:) = Smart_Features(:,4)';
%             Slope_comb_matrix(comb,:) = Smart_Features(:,1)';
%             Peak_comb_matrix(comb,:) = Smart_Features(:,2)';
%             Area_comb_matrix(comb,:) = Smart_Features(:,3)';
%            end % end for comb = 1:size(Channel_comb_matrix,1)
end    
%% Data Visualization

%     figure;
%     if use_conventional_features == 1
%         scatter_set = Conventional_Features;
%         no_epochs = round(size(scatter_set,1)/2);
%         subplot(1,2,1);
%         % 1. Scatter Plot - data_Set1
%         scolor =  [repmat([0 0 1],no_epochs,1);repmat([0.5 0.5 0.5],no_epochs,1)]; 
%         h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,scolor,'filled');
%         xlabel('Slope','FontSize',14);
%         ylabel('Mahalanobis','FontSize',14);
%         zlabel('AUC','FontSize',14);
%         title(['Fixed Interval Features, Window Length = ' num2str(wl*100) 'msec.'],'FontSize',14);
%         s = 0.5/wl;
%         %Obtain the axes size (in axpos) in Points
%         currentunits = get(gca,'Units');
%         set(gca, 'Units', 'Points');
%         axpos = get(gca,'Position');
%         set(gca, 'Units', currentunits);
%         markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
%         set(h, 'SizeData', markerWidth^8)
%     end
%     
%     if use_smart_features == 1
%         scatter_set = Smart_Features;
%         no_epochs = round(size(scatter_set,1)/2);
%         subplot(1,2,2);
%         % 2. Scatter Plot - data_Set2
%         scolor =  [repmat([0 0 1],no_epochs,1);repmat([0.5 0.5 0.5],no_epochs,1)]; 
%         h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,scolor,'filled');
%         xlabel('Slope','FontSize',14);
%         ylabel('Mahalanobis','FontSize',14);
%         zlabel('AUC','FontSize',14);
%         title('Variable Interval Features','FontSize',14);
%         s = 0.4;
%         %Obtain the axes size (in axpos) in Points
%         currentunits = get(gca,'Units');
%         set(gca, 'Units', 'Points');
%         axpos = get(gca,'Position');
%         set(gca, 'Units', currentunits);
%         markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
%         set(h, 'SizeData', markerWidth^16);
%     end
% %     %scatter_bad_trials = data_set2([bad_move_trials; (size(data_set2,1)/2)+bad_move_trials],:);
% %     scatter_bad_trials = data_set2(bad_move_trials,:);
% %     scatter_set = scatter_bad_trials;
% %     no_ep = size(scatter_set,1);
% %     subplot(1,2,2); hold on;
% %     % 1. Scatter Plot - data_Set1
% %     scolor =  repmat([1 0 0],no_ep,1); 
% %     h = scatter3(scatter_set(:,1),scatter_set(:,4),scatter_set(:,3),4,scolor,'filled');
% %     s = 0.4;
% %     %Obtain the axes size (in axpos) in Points
% %     currentunits = get(gca,'Units');
% %     set(gca, 'Units', 'Points');
% %     axpos = get(gca,'Position');
% %     set(gca, 'Units', currentunits);
% %     markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
% %     set(h, 'SizeData', markerWidth^16)
    
    
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
%% --------------------------------- Classifier Training and Cross-validation
    
    for use_feature = loop_start:loop_end
        if use_feature == 1     % Conventional Features
             good_trials_move_ch_avg = move_ch_avg;
             good_trials_rest_ch_avg = rest_ch_avg;
             apply_scaling_for_SVM = 0;                                                     % Do not apply scaling; improves accuracy. Added by Nikunj - 1/3/2015
             classifier_dataset = Conventional_Features; 
             test_Cov_Mat = conv_Cov_Mat;
             test_Mu_move = conv_Mu_move;
             
        elseif use_feature == 2 % Smart Features
            good_trials_move_ch_avg = move_ch_avg(good_move_trials,:);
            good_trials_rest_ch_avg = rest_ch_avg(good_move_trials,:);
            apply_scaling_for_SVM = 0;
            classifier_dataset = Smart_Features;
            test_Cov_Mat = smart_Cov_Mat;
            test_Mu_move = smart_Mu_move;
            
        end

        no_epochs = round(size(classifier_dataset,1)/2);
        
        %if use_previous_models == 1                                                                % Use cross validation trials from previous analysis - can't use it because some trials are rejected - 8/19/2015
            %CVO = prev_Performance.CVO;
        %else
            CVO = cvpartition(no_epochs,'kfold',10);
        %end
        data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)]; %Classifier labels 1 - Go, 2 - No-Go 

    % Support Vector Machines Classifier for EEG
    if use_svm_classifier == 1
        use_src_classifier = 0;
        % SVM-1. Scale data set between minval to maxval
        if  apply_scaling_for_SVM == 1
            [nr,nc] = size(classifier_dataset);
            scale_range = zeros(2,nc);
            for k = 1:nc
                attribute = classifier_dataset(:,k);
                scale_range(:,k) = [min(attribute); max(attribute)];
                attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
                classifier_dataset(:,k) = attribute;
            end
        end    
        % SVM-2. Create sparse matrix
        features_sparse = sparse(classifier_dataset);           
        % SVM-3. Hyperparameter Optimization (C,gamma) via Grid Search
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
    end
    
    % Sparse Reconstruction Classifier for EEG
    if use_src_classifier == 1
        use_svm_classifier = 0;  
        % Use classifier_dataset directly
        % SRC-1. Hyperparameter Optimization (sparsity level) via Grid Search
                for values = 5:1:30
                    fold_CM = [];
                    for cv_grid = 1:CVO.NumTestSets
                        grid_trIdx = CVO.training(cv_grid);
                        grid_teIdx = CVO.test(cv_grid);
                        
                        fold_test_labels = srclassifier(classifier_dataset([grid_trIdx;grid_trIdx],:)',classifier_dataset([grid_teIdx;grid_teIdx],:)',data_set_labels([grid_trIdx;grid_trIdx],1)','OMP',...
                                                        values);
                        fold_CM(:,:,cv_grid) = confusionmat(data_set_labels([grid_teIdx;grid_teIdx],1), fold_test_labels);                        
                    end
                    sparsity_levels(values) = sum(diag( sum(fold_CM,3)))/sum(sum( sum(fold_CM,3)));
                end
                [opt_sparsity_value(use_feature),opt_sparsity_level(use_feature)] = max(sparsity_levels);
    end
        
    % Train classifier model and cross-validate
        for cv_index = 1:CVO.NumTestSets
            trIdx = CVO.training(cv_index);
            teIdx = CVO.test(cv_index);
            
            % Do grid search here ----------------------------       
            
            if use_svm_classifier == 1
                eeg_svm_model{use_feature,cv_index} = ...
                    svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                    ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt(use_feature)) ' -g ' num2str(gamma_opt(use_feature)) ' -b 1']);
            end
            
            if use_src_classifier == 1
                % Not required to train SRC model. Directly perform
                % cross-validation. What??????             
            end
            
            if use_shifting_window_CV == 1
                % Sliding window validation 
                t1 = crossvalidation_trial_time(1);
                t2 = crossvalidation_trial_time(2);
                test_trials = [good_trials_move_ch_avg(teIdx==1,find(move_erp_time == t1):find(move_erp_time == t2));
                               good_trials_rest_ch_avg(teIdx==1,find(rest_erp_time == t1):find(rest_erp_time == t2))];                      
                test_trials_time = repmat(t1:1/Fs_eeg:t2,size(test_trials,1),1);
                test_trials_labels = data_set_labels([teIdx;teIdx],1);

                test_trial_decision = [];
                mean_test_trial_prob = [];
                test_trial_signal_window = [];
                test_trial_signal_time = [];
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
                    AUC_tt = zeros(size(test_trial_signal_window,1),1);
                    for ind1 = 1:size(test_trial_signal_window,1)
                        AUC_tt(ind1) = trapz(test_trial_signal_time(ind1,:),test_trial_signal_window(ind1,:));
                    end
                    test_trial_data_set = [test_trial_data_set AUC_tt];

                    %4. Mahalanobis distance of each trial from average over trials
                    test_mahal_dist = zeros(size(test_trial_signal_window,1),1);
                    for d = 1:size(test_trial_signal_window,1)
                        %test_mahal_dist(d) = sqrt(mahal(test_trial_signal_window(d,:),good_trials_move_ch_avg(trIdx,mlim1:mlim2)));
                        
                        % Direct Computation of Mahalanobis distance
                        x = test_trial_signal_window(d,:);
                        test_mahal_dist(d) = sqrt((x-test_Mu_move)/(test_Cov_Mat)*(x-test_Mu_move)');
                    end
                    test_trial_data_set = [test_trial_data_set test_mahal_dist];


                    % Apply SVM classifier
                    if use_svm_classifier == 1
                        % SVM-b1. Scale data set between minval to maxval
                        if  apply_scaling_for_SVM == 1
                        [nr,nc] = size(test_trial_data_set);
                        scale_range = zeros(2,nc);
                        for k = 1:nc
                            attribute = test_trial_data_set(:,k);
                            scale_range(:,k) = [min(attribute); max(attribute)];
                            attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
                            test_trial_data_set(:,k) = attribute;
                        end
                        end    
                        % SVM-b2. Create sparse matrix
                        test_trial_data_set_sparse = sparse(test_trial_data_set);    
                        % SVM-b3. SVM prediction
                        [internal_decision,internal_acc, internal_prob] = svmpredict(repmat(test_trials_labels(test_trial_no),size(test_trial_data_set_sparse,1),1), test_trial_data_set_sparse, eeg_svm_model{use_feature,cv_index}, '-b 1');
                        internal_decision(internal_prob(:,1) <= prob_est_threshold) = 2; %Should I use internal probability threshold?
                    end

                     % Apply SRC classifier
                     if use_src_classifier == 1
                        internal_decision = srclassifier(classifier_dataset',test_trial_data_set',data_set_labels','OMP',opt_sparsity_level(use_feature));
                        internal_prob = ones(length(internal_decision),1);
                     end
                                        
                     %-----------Find consecutive 1's >= Threshold
                     test_trial_decision(test_trial_no,:) = [0 0 0];
                     % Initially, assign mean of probabilities for 'rest'
                     % decisions. Change to mean of 'move' probabilities
                     % after consecutive counts threshold is exceeded.
                     mean_test_trial_prob(test_trial_no,:) = mean(internal_prob((internal_prob(:,1) <= prob_est_threshold),:),1); 
                     mycounter = 0;
                     for i = 2:length(internal_decision)
                         if (internal_decision(i)== 1) && (internal_decision(i-1)==1)
                             mycounter = mycounter+1;
                         else
                             mycounter = 0;
                         end
                         if mycounter >= consecutive_cnts_threshold - 1
                            test_trial_decision(test_trial_no,:) = [1 test_trial_signal_time(i,end) 0];
                            mean_test_trial_prob(test_trial_no,:) =  mean(internal_prob(i-mycounter:i,:),1);          % changed consecutive_cnts_threshold to mycounter, Nikunj - 6/28/14
                            break;
                         end
                     end

                     % Save information about number of signifcant
                     % maxima-minima
                     Max_Min_Thr = 1; % 3V
                     t2 = find(test_trials_time(1,:)==test_trial_decision(test_trial_no,2));
                     t1 = t2 - 1.5*Fs_eeg;  % Go back 1.5 sec
                     if t1 < 1
                            test_trial_decision(test_trial_no,3) = NaN; % Overflow
                     else
                        [xmax,indmax,xmin,indmin] = extrema(test_trials(test_trial_no,t1:t2));
                        ranges = abs(diff(test_trials(test_trial_no, sort([indmax indmin]))));
                        test_trial_decision(test_trial_no,3) = length(find(ranges >= Max_Min_Thr));
                     end

                end
                test_trial_decision(test_trial_decision(:,1) == 0,1) = 2;
                eeg_decision{use_feature,cv_index} = test_trial_decision(:,1);
                eeg_prob_estimates{use_feature,cv_index} = [mean_test_trial_prob test_trial_decision(:,2:3)]; % Time at which go was detected & No of significant abrupt changes
            else
                % Regular cross validation approach
                [eeg_decision{use_feature,cv_index}, accuracy, eeg_prob_estimates{use_feature,cv_index}] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), eeg_svm_model{use_feature,cv_index}, '-b 1');
            end

            eeg_CM(use_feature,:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),eeg_decision{use_feature,cv_index});
            eeg_sensitivity(use_feature,cv_index) = eeg_CM(use_feature,1,1,cv_index)/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index));
            eeg_specificity(use_feature,cv_index) = eeg_CM(use_feature,2,2,cv_index)/(eeg_CM(use_feature,2,2,cv_index)+eeg_CM(use_feature,2,1,cv_index));
            eeg_accur(use_feature,cv_index) = (eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index)+eeg_CM(use_feature,2,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))*100;   
        end % end crossvalidation for loop
        total_CM(use_feature,:,:) = sum(squeeze(eeg_CM(use_feature,:,:,:)),3);      
    end % end use_feature for loop   
    
% Save all optimization variables
All_optimized_channels{wl} = optimized_channels;
All_move_window(wl,:) = move_window;        % Only for Conventional Feature
All_rest_window(wl,:) = rest_window;        % Same for Conventional and Smart Feature
All_good_move_trials{wl} = good_move_trials;
All_bad_move_trials{wl} = bad_move_trials;
All_window_length_range = window_length_range;
All_Conventional_Features{wl} = Conventional_Features;
All_Smart_Features{wl} = Smart_Features;
All_conv_Cov_Mat{wl} = conv_Cov_Mat;
All_conv_Mu_move{wl} = conv_Mu_move;
All_smart_Cov_Mat{wl} = smart_Cov_Mat;
All_smart_Mu_move{wl} = smart_Mu_move;
if use_svm_classifier == 1
    All_C_opt(wl,:) = C_opt;
    All_gamma_opt(wl,:) = gamma_opt;
    All_eeg_svm_model{wl} = eeg_svm_model;
end

if use_src_classifier == 1
    All_opt_sparsity_value(wl,:) = opt_sparsity_value;
    All_opt_sparsity_level(wl,:) = opt_sparsity_level;
end
    
All_eeg_prob_estimates{wl} = eeg_prob_estimates;
All_eeg_decision{wl} = eeg_decision;
All_eeg_CM{wl} = eeg_CM;
All_total_CM{wl} = total_CM;
All_eeg_sensitivity{wl} = eeg_sensitivity;
All_eeg_specificity{wl} = eeg_specificity;
All_eeg_accur{wl} = eeg_accur;
end % end window_length for loop
%% Plot ROC Curve and determine optimal window length for smart and
if use_svm_classifier == 1
    if length(window_length_range) > 1
    for use_feature = loop_start:loop_end              
       % Choose optimal_wl
       for roc_wl = 1:length(window_length_range) 
           wl_CM = squeeze(All_total_CM{roc_wl}(use_feature,:,:));
           roc_measures(roc_wl,1:2,use_feature) = [wl_CM(1,1)/(wl_CM(1,1) + wl_CM(1,2)) ...  % TPR
                                wl_CM(2,1)/(wl_CM(2,1) + wl_CM(2,2))]; % FPR
           roc_measures(roc_wl,3,use_feature) = trace(wl_CM)/sum(sum(wl_CM));   % Accuracy
           my_prob_est = [];
           my_true_labels = [];
           X_Y_Thr = [];
           for kfold = 1:10
               each_fold_prob_est = All_eeg_prob_estimates{1,roc_wl}{use_feature,kfold}(:,1);     % Go Probability
               my_prob_est = [my_prob_est; each_fold_prob_est];
               my_true_labels = [my_true_labels; [ones(length(each_fold_prob_est)/2,1); 2*ones(length(each_fold_prob_est)/2,1)]];
           end
           [X_Y_Thr(:,1,use_feature),X_Y_Thr(:,2,use_feature),X_Y_Thr(:,3,use_feature),roc_OPT_AUC(roc_wl,3,use_feature),roc_OPT_AUC(roc_wl,1:2,use_feature)] = ...
               perfcurve(my_true_labels,my_prob_est,1);
           roc_X_Y_Thr{roc_wl,use_feature} = X_Y_Thr;
       end
    end
   figure; 
   
   if use_conventional_features == 1
       use_feature = 1;
       subplot(2,1,1);hold on; grid on;
       conv_fea_legend = plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','-b','LineWidth',2);
       plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','sb','MarkerSize',3,'LineWidth',2,'MarkerFaceColor',[0 0 1]);
       ylim([0 1]);
       ylabel('AUC','FontSize',10);
       xlabel('Window Length (sec.)','FontSize',10);
       % conv_optimal_window_length = input('Enter Optimal Window Length (Conventional Features): ');  % commented on 8/20/2015
       % conv_opt_wl_ind = find(window_length_range./Fs_eeg == conv_optimal_window_length);
       %[max_area_value,conv_opt_wl_ind] = max(round(roc_OPT_AUC(:,3,use_feature).*100)./100);
       max_auc = max(roc_OPT_AUC(:,3,use_feature));
       conv_opt_wl_ind = find(roc_OPT_AUC(:,3,use_feature) >= 0.95*max_auc,1);
       conv_optimal_window_length = window_length_range(conv_opt_wl_ind)/Fs_eeg;       
       plot(conv_optimal_window_length, roc_OPT_AUC(conv_opt_wl_ind,3,use_feature)','ok','MarkerSize',10,'LineWidth',2);

       subplot(2,1,2);hold on; grid on;
       %plot(roc_measures(:,2), roc_measures(:,1),'xk');
       %plot3(roc_measures(:,2), roc_measures(:,1),window_length_range','xk','MarkerSize',10,'LineWidth',2);
       h1 = plot(roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,1,use_feature),roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,2,use_feature),'b','LineWidth',2);
       plot(roc_measures(conv_opt_wl_ind,2,use_feature), roc_measures(conv_opt_wl_ind,1,use_feature),'xb','MarkerSize',10,'LineWidth',2);
       %text(roc_measures(conv_opt_wl_ind,2), roc_measures(conv_opt_wl_ind,1)-0.1,'p = 0.5','FontWeight','bold');

       plot(roc_OPT_AUC(conv_opt_wl_ind,1,use_feature), roc_OPT_AUC(conv_opt_wl_ind,2,use_feature),'xk','MarkerSize',10,'LineWidth',2);
       conv_p_opt_ind = find(roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,2,use_feature) == roc_OPT_AUC(conv_opt_wl_ind,2,use_feature),1,'first');
       text(roc_OPT_AUC(conv_opt_wl_ind,1,use_feature), roc_OPT_AUC(conv_opt_wl_ind,2,use_feature)+0.1,sprintf('p(opt) = %.2f',roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(conv_p_opt_ind,3,use_feature)),'FontWeight','bold'); % Change here           
       axis([0 1 0 1]);
       xlabel('FPR','FontSize',10);
       ylabel('TPR','FontSize',10);
       %title('ROC Curve','FontSize',12);
       legend(h1,'ROC Curve','Location','SouthEast');
       xlim([0 1]);
   end 
   if use_smart_features == 1
       use_feature = 2;
       subplot(2,1,1);hold on; grid on;
       smart_fea_legend = plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','-r','LineWidth',2);
       plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','sr','MarkerSize',3,'LineWidth',2,'MarkerFaceColor',[0 0 1]);
       ylim([0 1]);
       ylabel('AUC','FontSize',10);
       xlabel('Window Length (sec.)','FontSize',10);
       % smart_optimal_window_length = input('Enter Optimal Window Length (Smart Features): '); % commented on 8/20/15
       % smart_opt_wl_ind = find(window_length_range./Fs_eeg == smart_optimal_window_length);
       max_auc = max(roc_OPT_AUC(:,3,use_feature));
       smart_opt_wl_ind = find(roc_OPT_AUC(:,3,use_feature) >= 0.95*max_auc,1);
       smart_optimal_window_length = window_length_range(smart_opt_wl_ind)/Fs_eeg;       
       plot(smart_optimal_window_length, roc_OPT_AUC(smart_opt_wl_ind,3,use_feature)','ok','MarkerSize',10,'LineWidth',2);

       subplot(2,1,2);hold on; grid on;
       %plot(roc_measures(:,2), roc_measures(:,1),'xk');
       %plot3(roc_measures(:,2), roc_measures(:,1),window_length_range','xk','MarkerSize',10,'LineWidth',2);
       h1 = plot(roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,1,use_feature),roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,2,use_feature),'r','LineWidth',2);
       plot(roc_measures(smart_opt_wl_ind,2,use_feature), roc_measures(smart_opt_wl_ind,1,use_feature),'xr','MarkerSize',10,'LineWidth',2);
       %text(roc_measures(smart_opt_wl_ind,2), roc_measures(smart_opt_wl_ind,1)-0.1,'p = 0.5','FontWeight','bold');

       plot(roc_OPT_AUC(smart_opt_wl_ind,1,use_feature), roc_OPT_AUC(smart_opt_wl_ind,2,use_feature),'xk','MarkerSize',10,'LineWidth',2);
       smart_p_opt_ind = find(roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,2,use_feature) == roc_OPT_AUC(smart_opt_wl_ind,2,use_feature),1,'first');
       text(roc_OPT_AUC(smart_opt_wl_ind,1,use_feature), roc_OPT_AUC(smart_opt_wl_ind,2,use_feature)+0.1,sprintf('p(opt) = %.2f',roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(smart_p_opt_ind,3,use_feature)),'FontWeight','bold'); % Change here           
       axis([0 1 0 1]);
       xlabel('FPR','FontSize',10);
       ylabel('TPR','FontSize',10);
       %title('ROC Curve','FontSize',12);
       legend(h1,'ROC Curve','Location','SouthEast');
       xlim([0 1]);
       line([0 1],[0 1],'Color','k','LineWidth',1,'LineStyle','--');
   end  
   %export_fig 'MR_ses2_cond1_ROC_embs' '-png' '-transparent'
else
   conv_optimal_window_length = window_length_range./Fs_eeg;  
   conv_opt_wl_ind = 1; 
   wl_CM = squeeze(All_total_CM{conv_opt_wl_ind});
   roc_measures =   [wl_CM(1,1)/(wl_CM(1,1) + wl_CM(1,2)) ... % TPR
                     wl_CM(2,1)/(wl_CM(2,1) + wl_CM(2,2)) ... % FPR
                     trace(wl_CM)/sum(sum(wl_CM))]; 
   roc_OPT_AUC = []; 
   roc_X_Y_Thr = []; 
    end
end

if use_src_classifier == 1
    SVM_accuracy = prev_Performance.eeg_accur';
    SVM_opt_window_length = prev_Performance.smart_window_length;
    SRC_accuracies = [];
    for wl = 1:length(window_length_range)
        SRC_accuracies = [SRC_accuracies All_eeg_accur{wl}(2,:)'];
    end
    figure;
    group_labels = [window_length_range/Fs_eeg SVM_opt_window_length];
    boxplot([SRC_accuracies SVM_accuracy],'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','ko','colors','rb',...
                                                                                                                                     'positions',[1:length(window_length_range) length(window_length_range)+3],'labels',group_labels);
    ylim([0 110]);
    set(gca,'Ytick',[0:10:100],'Ygrid','on');
    xlabel('Window lengths (sec.)');
    ylabel('Classifier Accuracy (%)');
    title([Subject_name ', Cond ' num2str(Cond_num)]);
    src_smart_optimal_window_length = input('Enter SRC Optimal Window Length (Smart Features): ');
    src_smart_opt_wl_ind = find(window_length_range./Fs_eeg == src_smart_optimal_window_length);
    %fprintf('For window length = %.2f, sparsity level = %.2f \n',src_smart_optimal_window_length, All_opt_sparsity_level(src_smart_opt_wl_ind,use_feature));
    title([Subject_name ', Cond ' num2str(Cond_num) ', window length = ' num2str(src_smart_optimal_window_length)...
        ', sparsity level = ' num2str(All_opt_sparsity_level(src_smart_opt_wl_ind,use_feature))]);
    
    tiff_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_svm_src_comparison.tif'];
    fig_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_svm_src_comparison.fig'];
    print('-dtiff', '-r300', tiff_filename); 
    saveas(gcf,fig_filename);
    
    
end
%% Support Vector Machines Classifier for EMG
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
%% New Save Results - 2/12/2015
if use_svm_classifier == 1
    Performance = [];
    % Flags
    Performance.use_separate_move_rest_epochs = use_separate_move_rest_epochs;
    Performance.apply_epoch_standardization = apply_epoch_standardization;
    Performance.apply_baseline_correction = apply_baseline_correction;
    Performance.use_smart_features = use_smart_features;
    Performance.use_conventional_features = use_conventional_features;
    Performance.keep_bad_trials = keep_bad_trials;
    Performance.use_shifting_window_CV = use_shifting_window_CV;
    Performance.apply_scaling_for_SVM = apply_scaling_for_SVM;
    Performance.use_channel_optimization = use_channel_optimization;

    % Variables
    Performance.classchannels = classchannels;
    Performance.find_peak_interval = find_peak_interval;
    Performance.reject_trial_before = reject_trial_before;
    Performance.prob_est_threshold = prob_est_threshold;

    if use_conventional_features == 1
            Performance.conv_window_length = conv_optimal_window_length;        % +1 added by Nikunj, 5/14/2014, removed by Nikunj, 6/07/2014
            Performance.conv_opt_wl_ind = conv_opt_wl_ind;
            Performance.Conventional_Features = All_Conventional_Features{conv_opt_wl_ind};       % Old Features
    end

    if use_smart_features == 1
            Performance.opt_prob_threshold = roc_X_Y_Thr{smart_opt_wl_ind,2}(smart_p_opt_ind,3,2);
            Performance.move_window = []; % Not valid for Smart Features
            Performance.rest_window = All_rest_window(smart_opt_wl_ind,:);
            Performance.bad_move_trials = All_bad_move_trials{smart_opt_wl_ind};
            Performance.good_move_trials = All_good_move_trials{smart_opt_wl_ind};
            Performance.smart_window_length = smart_optimal_window_length;
            Performance.smart_opt_wl_ind = smart_opt_wl_ind;
            %Performance.data_set_labels = data_set_labels;         % Confusing. Better avoid saving it
            Performance.Smart_Features = All_Smart_Features{smart_opt_wl_ind};       % New Features
            Performance.C_opt = All_C_opt(smart_opt_wl_ind,2);
            Performance.gamma_opt = All_gamma_opt(smart_opt_wl_ind,2);
            Performance.smart_Cov_Mat = All_smart_Cov_Mat{smart_opt_wl_ind};
            Performance.smart_Mu_move = All_smart_Mu_move{smart_opt_wl_ind};
            Performance.eeg_svm_model = All_eeg_svm_model{smart_opt_wl_ind}(2,:);   % Only for Smart Features
            Performance.eeg_prob_estimates = All_eeg_prob_estimates{smart_opt_wl_ind}(2,:);
            Performance.eeg_decision = All_eeg_decision{smart_opt_wl_ind}(2,:);
            Performance.eeg_CM = squeeze(All_eeg_CM{smart_opt_wl_ind}(2,:,:,:));
            Performance.eeg_sensitivity = All_eeg_sensitivity{smart_opt_wl_ind}(2,:);
            Performance.eeg_specificity = All_eeg_specificity{smart_opt_wl_ind}(2,:);
            Performance.eeg_accur = All_eeg_accur{smart_opt_wl_ind}(2,:);
            Performance.optimized_channels = All_optimized_channels{smart_opt_wl_ind};  %test_change
    end

    % All variables used during optimization
    Performance.consecutive_cnts_threshold = consecutive_cnts_threshold;
    Performance.Max_Min_Thr = Max_Min_Thr;
    Performance.CVO = CVO;
    Performance.roc_measures = roc_measures;
    Performance.roc_X_Y_Thr = roc_X_Y_Thr;
    Performance.roc_OPT_AUC = roc_OPT_AUC;

    Performance.All_window_length_range = All_window_length_range;
    Performance.All_total_CM = All_total_CM;
    Performance.All_move_window = All_move_window;
    Performance.All_rest_window = All_rest_window;
    Performance.All_good_move_trials = All_good_move_trials;
    Performance.All_bad_move_trials = All_bad_move_trials;
    Performance.All_Conventional_Features = All_Conventional_Features;
    Performance.All_Smart_Features = All_Smart_Features;
    Performance.All_conv_Cov_Mat =  All_conv_Cov_Mat;
    Performance.All_conv_Mu_move = All_conv_Mu_move;
    Performance.All_smart_Cov_Mat = All_smart_Cov_Mat;
    Performance.All_smart_Mu_move = All_smart_Mu_move;

    Performance.All_C_opt = All_C_opt;
    Performance.All_gamma_opt = All_gamma_opt;
    Performance.All_eeg_svm_model = All_eeg_svm_model;
    Performance.All_eeg_prob_estimates = All_eeg_prob_estimates;
    Performance.All_eeg_decision = All_eeg_decision;
    Performance.All_eeg_CM = All_eeg_CM;
    Performance.All_eeg_sensitivity = All_eeg_sensitivity;
    Performance.All_eeg_specificity = All_eeg_specificity;
    Performance.All_eeg_accur = All_eeg_accur;
    Performance.All_optimized_channels = All_optimized_channels;  %test_change
    
    % Miscellaneous
    %Performance.test_trials_time = test_trials_time;
    %Performance.test_trial_signal_time =  test_trial_signal_time;

    file_identifier = [];
    if use_conventional_features == 1
        file_identifier = [ file_identifier '_conventional'];
    end
    if use_smart_features == 1
        file_identifier = [ file_identifier '_smart'];
    end
    % if use_shifting_window_CV == 1
    %     file_identifier = [ file_identifier '_online'];
    % else
    %     file_identifier = [ file_identifier '_offline'];
    % end
    if use_channel_optimization == 0
        file_identifier = [ file_identifier '_manual'];
    end

    filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized' file_identifier datestr(now,'_dd_mm_yy_HHMM') '.mat']; 
    save(filename2,'Performance');   
end

if use_src_classifier == 1
    SRC_Performance = [];
    SRC_Performance.use_separate_move_rest_epochs = use_separate_move_rest_epochs;
    SRC_Performance.apply_epoch_standardization = apply_epoch_standardization;
    SRC_Performance.apply_baseline_correction = apply_baseline_correction;
    SRC_Performance.use_smart_features = use_smart_features;
    SRC_Performance.use_conventional_features = use_conventional_features;
    SRC_Performance.keep_bad_trials = keep_bad_trials;
    SRC_Performance.use_shifting_window_CV = use_shifting_window_CV;
    
    % Variables
    SRC_Performance.classchannels = classchannels;
    SRC_Performance.find_peak_interval = find_peak_interval;
    SRC_Performance.reject_trial_before = reject_trial_before;
    SRC_Performance.prob_est_threshold = prob_est_threshold;

    if use_conventional_features == 1
            SRC_Performance.conv_window_length = conv_optimal_window_length;        % +1 added by Nikunj, 5/14/2014, removed by Nikunj, 6/07/2014
            SRC_Performance.conv_opt_wl_ind = conv_opt_wl_ind;
            SRC_Performance.Conventional_Features = All_Conventional_Features{conv_opt_wl_ind};       % Old Features
    end

    if use_smart_features == 1
            %SRC_Performance.opt_prob_threshold = roc_X_Y_Thr{smart_opt_wl_ind,2}(smart_p_opt_ind,3,2);
            SRC_Performance.move_window = []; % Not valid for Smart Features
            SRC_Performance.rest_window = All_rest_window(src_smart_opt_wl_ind,:);
            SRC_Performance.bad_move_trials = All_bad_move_trials{src_smart_opt_wl_ind};
            SRC_Performance.good_move_trials = All_good_move_trials{src_smart_opt_wl_ind};
            SRC_Performance.src_smart_window_length = src_smart_optimal_window_length;
            SRC_Performance.src_smart_opt_wl_ind = src_smart_opt_wl_ind;
            %Performance.data_set_labels = data_set_labels;         % Confusing. Better avoid saving it
            SRC_Performance.opt_sparsity_level = All_opt_sparsity_level(src_smart_opt_wl_ind,2);
            SRC_Performance.Smart_Features = All_Smart_Features{src_smart_opt_wl_ind};       % New Features
            SRC_Performance.smart_Cov_Mat = All_smart_Cov_Mat{src_smart_opt_wl_ind};
            SRC_Performance.smart_Mu_move = All_smart_Mu_move{src_smart_opt_wl_ind};
            SRC_Performance.eeg_prob_estimates = All_eeg_prob_estimates{src_smart_opt_wl_ind}(2,:);
            SRC_Performance.eeg_decision = All_eeg_decision{src_smart_opt_wl_ind}(2,:);
            SRC_Performance.eeg_CM = squeeze(All_eeg_CM{src_smart_opt_wl_ind}(2,:,:,:));
            SRC_Performance.eeg_sensitivity = All_eeg_sensitivity{src_smart_opt_wl_ind}(2,:);
            SRC_Performance.eeg_specificity = All_eeg_specificity{src_smart_opt_wl_ind}(2,:);
            SRC_Performance.eeg_accur = All_eeg_accur{src_smart_opt_wl_ind}(2,:);
    end
    
     % All variables used during optimization
    SRC_Performance.consecutive_cnts_threshold = consecutive_cnts_threshold;
    SRC_Performance.Max_Min_Thr = Max_Min_Thr;
    SRC_Performance.CVO = CVO;
%     Performance.roc_measures = roc_measures;
%     Performance.roc_X_Y_Thr = roc_X_Y_Thr;
%     Performance.roc_OPT_AUC = roc_OPT_AUC;

    SRC_Performance.All_opt_sparsity_level = All_opt_sparsity_level;
    SRC_Performance.All_opt_sparsity_value = All_opt_sparsity_value;    
    SRC_Performance.All_window_length_range = All_window_length_range;
    SRC_Performance.All_total_CM = All_total_CM;
    SRC_Performance.All_move_window = All_move_window;
    SRC_Performance.All_rest_window = All_rest_window;
    SRC_Performance.All_good_move_trials = All_good_move_trials;
    SRC_Performance.All_bad_move_trials = All_bad_move_trials;
    SRC_Performance.All_Conventional_Features = All_Conventional_Features;
    SRC_Performance.All_Smart_Features = All_Smart_Features;
    SRC_Performance.All_conv_Cov_Mat =  All_conv_Cov_Mat;
    SRC_Performance.All_conv_Mu_move = All_conv_Mu_move;
    SRC_Performance.All_smart_Cov_Mat = All_smart_Cov_Mat;
    SRC_Performance.All_smart_Mu_move = All_smart_Mu_move;

    SRC_Performance.All_eeg_prob_estimates = All_eeg_prob_estimates;
    SRC_Performance.All_eeg_decision = All_eeg_decision;
    SRC_Performance.All_eeg_CM = All_eeg_CM;
    SRC_Performance.All_eeg_sensitivity = All_eeg_sensitivity;
    SRC_Performance.All_eeg_specificity = All_eeg_specificity;
    SRC_Performance.All_eeg_accur = All_eeg_accur;
    
     file_identifier = [];
    if use_conventional_features == 1
        file_identifier = [ file_identifier '_conventional'];
    end
    if use_smart_features == 1
        file_identifier = [ file_identifier '_smart'];
    end
    
    filename3 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_src_performance_optimized' file_identifier '.mat']; %datestr(now,'dd_mm_yy_HHMM')
    save(filename3,'SRC_Performance');   
end
%% Plot sensitivity & specificity
    if use_svm_classifier == 1
        if use_previous_models == 1
            figure; 
            group_names = {'Fixed','Variable', 'Variable old'};
            online_fixed = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);
            online_variable = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
            %online_fixed_old = prev_Performance.All_eeg_accur{prev_Performance.conv_opt_wl_ind}(1,:);
            online_variable_old = prev_Performance.All_eeg_accur{prev_Performance.smart_opt_wl_ind}(2,:);

            % filename3 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_smart_offline_nic.mat'];
            % load(filename3);
            % offline_fixed = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);


            h = boxplot([online_fixed' online_variable' online_variable_old'] ,'labels',group_names,'widths',0.5);
             set(h,'LineWidth',2);
            v = axis;
            axis([v(1) v(2) 0 110]);
            ylabel('Classification Accuracy (%)','FontSize',12);
            title([Subject_name ', Mode ' num2str(Cond_num)],'FontSize',12);

            %export_fig 'TA_ses1_cond3_smart_conv_comparison' '-png' '-transparent'

            % if use_shifting_window_CV == 1
            %     title('Online Simulation','FontSize',12);
            %     %mtit('Online Simulation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
            % else
            %     title('Offline Validation','FontSize',12);
            %     %mtit('Offline Validation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
            % end

            p_values = [];
            % both-tailed Wilcoxon Rank sum Test, i.e. median(day 4) >< median(day 5)
            [pwilcoxon,h,stats] = ranksum(online_variable,online_fixed,'alpha',0.05,'tail','right');
            if (pwilcoxon <= 0.05) 
                p_values = [p_values 0.05];
            else
                p_values = [p_values NaN];
            end
            [pwilcoxon,h,stats] = ranksum(online_variable,online_variable_old,'alpha',0.05,'tail','right');
            if (pwilcoxon <= 0.05) 
                p_values = [p_values 0.05];
            else
                p_values = [p_values NaN];
            end
            sigstar({[1 2],[2 3]},p_values);
        else
            figure; 
            group_names = {{'Fixed';'Adaptive'},{'window';'window'}};
            online_variable = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
            if use_conventional_features == 1
                online_fixed = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);
            else
                online_fixed = zeros(size(online_variable));
            end
            
            h = boxplot([online_fixed' online_variable'] ,'labels',group_names,'widths',0.5);
             set(h,'LineWidth',2);
            v = axis;
            axis([v(1) v(2) 0 110]);
            ylabel('Classification Accuracy (%)','FontSize',12);
            title([Subject_name ', Mode ' num2str(Cond_num)],'FontSize',12);

            %export_fig 'TA_ses1_cond3_smart_conv_comparison' '-png' '-transparent'

            % if use_shifting_window_CV == 1
            %     title('Online Simulation','FontSize',12);
            %     %mtit('Online Simulation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
            % else
            %     title('Offline Validation','FontSize',12);
            %     %mtit('Offline Validation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
            % end

            p_values = [];
            % both-tailed Wilcoxon Rank sum Test, i.e. median(day 4) >< median(day 5)
            [pwilcoxon,h,stats] = ranksum(online_variable,online_fixed,'alpha',0.05,'tail','right');
            if (pwilcoxon <= 0.05) 
                p_values = [p_values 0.05];
            else
                p_values = [p_values NaN];
            end
            sigstar({[1 2]},p_values);
        end
    end
elseif regular_or_chance_level_classifier == 0
% Estimate chance level performance by permutation of labels and data
    for use_feature = loop_start:loop_end
        if use_feature == 1     % Conventional Features
             good_trials_move_ch_avg = move_ch_avg;
             good_trials_rest_ch_avg = rest_ch_avg;
             apply_scaling_for_SVM = 0;                                                     % Do not apply scaling; improves accuracy. Added by Nikunj - 1/3/2015
             classifier_dataset = prev_Performance.Conventional_Features; 
             test_Cov_Mat = prev_Performance.All_conv_Cov_Mat{prev_Performance.conv_opt_wl_ind};
             test_Mu_move = prev_Performance.All_conv_Mu_move{prev_Performance.conv_opt_wl_ind};
             window_length = round(prev_Performance.conv_window_length*Fs_eeg);
             
        elseif use_feature == 2 % Smart Features
            good_move_trials = prev_Performance.good_move_trials;
            good_trials_move_ch_avg = move_ch_avg(good_move_trials,:);
            good_trials_rest_ch_avg = rest_ch_avg(good_move_trials,:);
            apply_scaling_for_SVM = 0;
            classifier_dataset = prev_Performance.Smart_Features;
            test_Cov_Mat = prev_Performance.smart_Cov_Mat;
            test_Mu_move = prev_Performance.smart_Mu_move;
            window_length = round(prev_Performance.smart_window_length*Fs_eeg);
            
        end
        original_dataset = classifier_dataset;
        num_permutations = 1000;
        class_labels_permutations = zeros(size(classifier_dataset,1),num_permutations);
        Chance_accuracy_permutations_labels = zeros(num_permutations,1);
        Chance_cv_accur_permutations_labels = zeros(num_permutations,10);
        
        Chance_accuracy_permutations_data = zeros(num_permutations,1);
        Chance_cv_accur_permutations_data = zeros(num_permutations,10);
        
        for M1 = 1:num_permutations % Number of permutations
               class_labels_permutations(:,M1) =  randperm(size(classifier_dataset,1));
               classifier_dataset = classifier_dataset(class_labels_permutations(:,M1),:);
               no_epochs = round(size(classifier_dataset,1)/2);        
               CVO = cvpartition(no_epochs,'kfold',10);
               data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)]; %Classifier labels 1 - Go, 2 - No-Go 

            % 1. Scale data set between minval to maxval
                if  apply_scaling_for_SVM == 1
                    [nr,nc] = size(classifier_dataset);
                    scale_range = zeros(2,nc);
                    for k = 1:nc
                        attribute = classifier_dataset(:,k);
                        scale_range(:,k) = [min(attribute); max(attribute)];
                        attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
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
                for cv_index = 1:CVO.NumTestSets
                    trIdx = CVO.training(cv_index);
                    teIdx = CVO.test(cv_index);

                     eeg_svm_model{use_feature,cv_index} = ...
                         svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                         ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt(use_feature)) ' -g ' num2str(gamma_opt(use_feature)) ' -b 1']);

                    if use_shifting_window_CV == 1
                        % Sliding window validation 
                        t1 = crossvalidation_trial_time(1);
                        t2 = crossvalidation_trial_time(2);
                        test_trials = [good_trials_move_ch_avg(teIdx==1,find(move_erp_time == t1):find(move_erp_time == t2));
                                       good_trials_rest_ch_avg(teIdx==1,find(rest_erp_time == t1):find(rest_erp_time == t2))];                      
                        test_trials_time = repmat(t1:1/Fs_eeg:t2,size(test_trials,1),1);
                        test_trials_labels = data_set_labels([teIdx;teIdx],1);

                        test_trial_decision = [];
                        mean_test_trial_prob = [];
                        test_trial_signal_window = [];
                        test_trial_signal_time = [];
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
                            AUC_tt = zeros(size(test_trial_signal_window,1),1);
                            for ind1 = 1:size(test_trial_signal_window,1)
                                AUC_tt(ind1) = trapz(test_trial_signal_time(ind1,:),test_trial_signal_window(ind1,:));
                            end
                            test_trial_data_set = [test_trial_data_set AUC_tt];

                            %4. Mahalanobis distance of each trial from average over trials
                            test_mahal_dist = zeros(size(test_trial_signal_window,1),1);
                            for d = 1:size(test_trial_signal_window,1)
                                %test_mahal_dist(d) = sqrt(mahal(test_trial_signal_window(d,:),good_trials_move_ch_avg(trIdx,mlim1:mlim2)));

                                % Direct Computation of Mahalanobis distance
                                x = test_trial_signal_window(d,:);
                                test_mahal_dist(d) = sqrt((x-test_Mu_move)/(test_Cov_Mat)*(x-test_Mu_move)');
                            end
                            test_trial_data_set = [test_trial_data_set test_mahal_dist];


                            % Apply SVM classifier
                            % T1. Scale data set between minval to maxval
                            if  apply_scaling_for_SVM == 1
                                [nr,nc] = size(test_trial_data_set);
                                scale_range = zeros(2,nc);
                                for k = 1:nc
                                    attribute = test_trial_data_set(:,k);
                                    scale_range(:,k) = [min(attribute); max(attribute)];
                                    attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
                                    test_trial_data_set(:,k) = attribute;
                                end
                            end    
                            % T2. Create sparse matrix
                             test_trial_data_set = sparse(test_trial_data_set);    
                            % T3. SVM prediction
                             [internal_decision,internal_acc, internal_prob] = svmpredict(repmat(test_trials_labels(test_trial_no),size(test_trial_data_set,1),1), test_trial_data_set, eeg_svm_model{use_feature,cv_index}, '-b 1');
                             internal_decision(internal_prob(:,1) <= prob_est_threshold) = 2; %Should I use internal probability threshold? 

                             %-----------Find consecutive 1's >= Threshold
                             test_trial_decision(test_trial_no,:) = [0 0 0];
                             % Initially, assign mean of probabilities for 'rest'
                             % decisions. Change to mean of 'move' probabilities
                             % after consecutive counts threshold is exceeded.
                             mean_test_trial_prob(test_trial_no,:) = mean(internal_prob((internal_prob(:,1) <= prob_est_threshold),:),1); 
                             mycounter = 0;
                             for i = 2:length(internal_decision)
                                 if (internal_decision(i)== 1) && (internal_decision(i-1)==1)
                                     mycounter = mycounter+1;
                                 else
                                     mycounter = 0;
                                 end
                                 if mycounter >= consecutive_cnts_threshold - 1
                                    test_trial_decision(test_trial_no,:) = [1 test_trial_signal_time(i,end) 0];
                                    mean_test_trial_prob(test_trial_no,:) =  mean(internal_prob(i-mycounter:i,:),1);          % changed consecutive_cnts_threshold to mycounter, Nikunj - 6/28/14
                                    break;
                                 end
                             end

                             % Save information about number of signifcant
                             % maxima-minima
                             Max_Min_Thr = 1; % 3V
                             t2 = find(test_trials_time(1,:)==test_trial_decision(test_trial_no,2));
                             t1 = t2 - 1.5*Fs_eeg;  % Go back 1.5 sec
                             if t1 < 1
                                    test_trial_decision(test_trial_no,3) = NaN; % Overflow
                             else
                                [xmax,indmax,xmin,indmin] = extrema(test_trials(test_trial_no,t1:t2));
                                ranges = abs(diff(test_trials(test_trial_no, sort([indmax indmin]))));
                                test_trial_decision(test_trial_no,3) = length(find(ranges >= Max_Min_Thr));
                             end

                        end
                        test_trial_decision(test_trial_decision(:,1) == 0,1) = 2;
                        eeg_decision{use_feature,cv_index} = test_trial_decision(:,1);
                        eeg_prob_estimates{use_feature,cv_index} = [mean_test_trial_prob test_trial_decision(:,2:3)]; % Time at which go was detected & No of significant abrupt changes
                    else
                        % Regular cross validation approach
                        [eeg_decision{use_feature,cv_index}, accuracy, eeg_prob_estimates{use_feature,cv_index}] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), eeg_svm_model{use_feature,cv_index}, '-b 1');
                   end

                    eeg_CM(use_feature,:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),eeg_decision{use_feature,cv_index});
                    eeg_sensitivity(use_feature,cv_index) = eeg_CM(use_feature,1,1,cv_index)/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index));
                    eeg_specificity(use_feature,cv_index) = eeg_CM(use_feature,2,2,cv_index)/(eeg_CM(use_feature,2,2,cv_index)+eeg_CM(use_feature,2,1,cv_index));
                    eeg_accur(use_feature,cv_index) = (eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index)+eeg_CM(use_feature,2,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))*100;   
                end % end crossvalidation for loop
            
            % 5. Save values of chance level performance
            Chance_accuracy_permutations_labels(M1) = mean(eeg_accur(use_feature,:));
            Chance_cv_accur_permutations_labels(M1,:) = eeg_accur(use_feature,:);
            
        end % end permutations (M1) loop
        
        data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)]; %Classifier labels 1 - Go, 2 - No-Go 
        for M2 = 1:num_permutations % Number of permutations
               classifier_dataset = original_dataset;
               [tot_trials,tot_features] = size(classifier_dataset);
               class1_dataset = classifier_dataset(1:tot_trials/2,:);
               class2_dataset = classifier_dataset(tot_trials/2 + 1:tot_trials,:);
               classifier_dataset = [shake(class1_dataset,1);
                                                     shake(class2_dataset,1)];
               
               no_epochs = round(size(classifier_dataset,1)/2);        
               CVO = cvpartition(no_epochs,'kfold',10);
               

            % 1. Scale data set between minval to maxval
                if  apply_scaling_for_SVM == 1
                    [nr,nc] = size(classifier_dataset);
                    scale_range = zeros(2,nc);
                    for k = 1:nc
                        attribute = classifier_dataset(:,k);
                        scale_range(:,k) = [min(attribute); max(attribute)];
                        attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
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
                for cv_index = 1:CVO.NumTestSets
                    trIdx = CVO.training(cv_index);
                    teIdx = CVO.test(cv_index);

                     eeg_svm_model{use_feature,cv_index} = ...
                         svmtrain(data_set_labels([trIdx;trIdx],1), features_sparse([trIdx;trIdx],:),...
                         ['-t ' num2str(kernel_type) ' -c ' num2str(C_opt(use_feature)) ' -g ' num2str(gamma_opt(use_feature)) ' -b 1']);

                    if use_shifting_window_CV == 1
                        % Sliding window validation 
                        t1 = crossvalidation_trial_time(1);
                        t2 = crossvalidation_trial_time(2);
                        test_trials = [good_trials_move_ch_avg(teIdx==1,find(move_erp_time == t1):find(move_erp_time == t2));
                                       good_trials_rest_ch_avg(teIdx==1,find(rest_erp_time == t1):find(rest_erp_time == t2))];                      
                        test_trials_time = repmat(t1:1/Fs_eeg:t2,size(test_trials,1),1);
                        test_trials_labels = data_set_labels([teIdx;teIdx],1);

                        test_trial_decision = [];
                        mean_test_trial_prob = [];
                        test_trial_signal_window = [];
                        test_trial_signal_time = [];
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
                            AUC_tt = zeros(size(test_trial_signal_window,1),1);
                            for ind1 = 1:size(test_trial_signal_window,1)
                                AUC_tt(ind1) = trapz(test_trial_signal_time(ind1,:),test_trial_signal_window(ind1,:));
                            end
                            test_trial_data_set = [test_trial_data_set AUC_tt];

                            %4. Mahalanobis distance of each trial from average over trials
                            test_mahal_dist = zeros(size(test_trial_signal_window,1),1);
                            for d = 1:size(test_trial_signal_window,1)
                                %test_mahal_dist(d) = sqrt(mahal(test_trial_signal_window(d,:),good_trials_move_ch_avg(trIdx,mlim1:mlim2)));

                                % Direct Computation of Mahalanobis distance
                                x = test_trial_signal_window(d,:);
                                test_mahal_dist(d) = sqrt((x-test_Mu_move)/(test_Cov_Mat)*(x-test_Mu_move)');
                            end
                            test_trial_data_set = [test_trial_data_set test_mahal_dist];


                            % Apply SVM classifier
                            % T1. Scale data set between minval to maxval
                            if  apply_scaling_for_SVM == 1
                                [nr,nc] = size(test_trial_data_set);
                                scale_range = zeros(2,nc);
                                for k = 1:nc
                                    attribute = test_trial_data_set(:,k);
                                    scale_range(:,k) = [min(attribute); max(attribute)];
                                    attribute = ((attribute - scale_range(1,k))./((scale_range(2,k) - scale_range(1,k))))*(maxval_SVM_scaling - minval_SVM_scaling) + minval_SVM_scaling;
                                    test_trial_data_set(:,k) = attribute;
                                end
                            end    
                            % T2. Create sparse matrix
                             test_trial_data_set = sparse(test_trial_data_set);    
                            % T3. SVM prediction
                             [internal_decision,internal_acc, internal_prob] = svmpredict(repmat(test_trials_labels(test_trial_no),size(test_trial_data_set,1),1), test_trial_data_set, eeg_svm_model{use_feature,cv_index}, '-b 1');
                             internal_decision(internal_prob(:,1) <= prob_est_threshold) = 2; %Should I use internal probability threshold? 

                             %-----------Find consecutive 1's >= Threshold
                             test_trial_decision(test_trial_no,:) = [0 0 0];
                             % Initially, assign mean of probabilities for 'rest'
                             % decisions. Change to mean of 'move' probabilities
                             % after consecutive counts threshold is exceeded.
                             mean_test_trial_prob(test_trial_no,:) = mean(internal_prob((internal_prob(:,1) <= prob_est_threshold),:),1); 
                             mycounter = 0;
                             for i = 2:length(internal_decision)
                                 if (internal_decision(i)== 1) && (internal_decision(i-1)==1)
                                     mycounter = mycounter+1;
                                 else
                                     mycounter = 0;
                                 end
                                 if mycounter >= consecutive_cnts_threshold - 1
                                    test_trial_decision(test_trial_no,:) = [1 test_trial_signal_time(i,end) 0];
                                    mean_test_trial_prob(test_trial_no,:) =  mean(internal_prob(i-mycounter:i,:),1);          % changed consecutive_cnts_threshold to mycounter, Nikunj - 6/28/14
                                    break;
                                 end
                             end

                             % Save information about number of signifcant
                             % maxima-minima
                             Max_Min_Thr = 1; % 3V
                             t2 = find(test_trials_time(1,:)==test_trial_decision(test_trial_no,2));
                             t1 = t2 - 1.5*Fs_eeg;  % Go back 1.5 sec
                             if t1 < 1
                                    test_trial_decision(test_trial_no,3) = NaN; % Overflow
                             else
                                [xmax,indmax,xmin,indmin] = extrema(test_trials(test_trial_no,t1:t2));
                                ranges = abs(diff(test_trials(test_trial_no, sort([indmax indmin]))));
                                test_trial_decision(test_trial_no,3) = length(find(ranges >= Max_Min_Thr));
                             end

                        end
                        test_trial_decision(test_trial_decision(:,1) == 0,1) = 2;
                        eeg_decision{use_feature,cv_index} = test_trial_decision(:,1);
                        eeg_prob_estimates{use_feature,cv_index} = [mean_test_trial_prob test_trial_decision(:,2:3)]; % Time at which go was detected & No of significant abrupt changes
                    else
                        % Regular cross validation approach
                        [eeg_decision{use_feature,cv_index}, accuracy, eeg_prob_estimates{use_feature,cv_index}] = svmpredict(data_set_labels([teIdx;teIdx],1), features_sparse([teIdx;teIdx],:), eeg_svm_model{use_feature,cv_index}, '-b 1');
                   end

                    eeg_CM(use_feature,:,:,cv_index) = confusionmat(data_set_labels([teIdx;teIdx],1),eeg_decision{use_feature,cv_index});
                    eeg_sensitivity(use_feature,cv_index) = eeg_CM(use_feature,1,1,cv_index)/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index));
                    eeg_specificity(use_feature,cv_index) = eeg_CM(use_feature,2,2,cv_index)/(eeg_CM(use_feature,2,2,cv_index)+eeg_CM(use_feature,2,1,cv_index));
                    eeg_accur(use_feature,cv_index) = (eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))/(eeg_CM(use_feature,1,1,cv_index)+eeg_CM(use_feature,1,2,cv_index)+eeg_CM(use_feature,2,1,cv_index)+eeg_CM(use_feature,2,2,cv_index))*100;   
                end % end crossvalidation for loop
            
            % 5. Save values of chance level performance
            Chance_accuracy_permutations_data(M2) = mean(eeg_accur(use_feature,:));
            Chance_cv_accur_permutations_data(M2,:) = eeg_accur(use_feature,:);
            
        end % end permutations (M2) loop
        
    end % end use_feature for loop

    file_identifier = [];
    if use_conventional_features == 1
        file_identifier = [ file_identifier '_conventional'];
    end
    if use_smart_features == 1
        file_identifier = [ file_identifier '_smart'];
    end
    % if use_shifting_window_CV == 1
    %     file_identifier = [ file_identifier '_online'];
    % else
    %     file_identifier = [ file_identifier '_offline'];
    % end

    chance_level_permuted_labels_errors =  1 - (Chance_accuracy_permutations_labels./100);
    chance_level_permuted_data_errors =  1 - (Chance_accuracy_permutations_data./100);
    [Pcdf_labels,errors_labels] = ecdf(chance_level_permuted_labels_errors);
    [Pcdf_data,errors_data] = ecdf(chance_level_permuted_data_errors);

    figure; plot(errors_labels,Pcdf_labels);
    axis([0 1 0 1]);
    hold on; plot(errors_data,Pcdf_data,'r');
    line([classification_error classification_error],[0 1],'Color',[0 0 0])

    p_data_m1 = Pcdf_data(find(errors_data <= classification_error,1,'last'));
    p_labels_m1 = Pcdf_labels(find(errors_labels <= classification_error,1,'last'));
    if isempty(p_data_m1)
        p_data_m1 = 0.0001;
    end

    if isempty(p_labels_m1)
        p_labels_m1 = 0.0001;
    end
    p_data_m2 = (length(find(chance_level_permuted_data_errors <= classification_error)) + 1 )/ (num_permutations + 1);
    p_labels_m2 = (length(find(chance_level_permuted_labels_errors <= classification_error)) + 1 )/ (num_permutations + 1);
    
    filename3 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_chance_level' file_identifier '2.mat']; %datestr(now,'dd_mm_yy_HHMM')
    save(filename3,'Chance_accuracy_permutations_labels','Chance_cv_accur_permutations_labels','class_labels_permutations',...
                                  'Chance_accuracy_permutations_data','Chance_cv_accur_permutations_data','p_data_m1','p_data_m2','p_labels_m1','p_labels_m2');        

    
end
%% Old Save Results
% Performance = [];
% % Flags
% %Performance.use_mahalanobis = 1;    % Always use Mahalanobis Distance
% Performance.use_separate_move_rest_epochs = use_separate_move_rest_epochs;
% Performance.apply_epoch_standardization = apply_epoch_standardization;
% Performance.apply_baseline_correction = apply_baseline_correction;
% Performance.use_smart_features = use_smart_features;
% Performance.use_conventional_features = use_conventional_features;
% Performance.keep_bad_trials = keep_bad_trials;
% Performance.use_shifting_window_CV = use_shifting_window_CV;
% Performance.apply_scaling_for_SVM = apply_scaling_for_SVM;
% 
% % Variables
% Performance.classchannels = classchannels;
% Performance.find_peak_interval = find_peak_interval;
% Performance.reject_trial_before = reject_trial_before;
% Performance.move_window = move_window;
% Performance.rest_window = rest_window;
% Performance.bad_move_trials = bad_move_trials;
% Performance.good_move_trials = good_move_trials;
% Performance.window_length = window_length+1;        % Correction added by Nikunj, 5/14/2014
% 
% %Performance.move_ch_avg = move_ch_avg;
% Performance.data_set_labels = data_set_labels;
% Performance.Conventional_Features = Conventional_Features;       % Old Features
% Performance.Smart_Features = Smart_Features;       % New Features
% % Performance.emg_data_set = emg_data_set;
% 
% Performance.C_opt = C_opt;
% Performance.gamma_opt = gamma_opt;
% % Performance.emg_C_opt = emg_C_opt;
% % Performance.emg_gamma_opt = emg_gamma_opt;
% Performance.consecutive_cnts_threshold = consecutive_cnts_threshold;
% Performance.prob_est_threshold = prob_est_threshold;
% Performance.Max_Min_Thr = Max_Min_Thr;
% 
% Performance.CVO = CVO;
% % Performance.lda_sensitivity = lda_sensitivity;
% % Performance.lda_specificity = lda_specificity;
% % Performance.lda_accur = lda_accur;
% % Performance.svm_sensitivity = svm_sensitivity;
% % Performance.svm_specificity = svm_specificity;
% % Performance.svm_accur = svm_accur;
% 
% Performance.eeg_svm_model = eeg_svm_model;
% % Performance.emg_svm_model = emg_svm_model;
% 
% Performance.eeg_prob_estimates = eeg_prob_estimates;
% % Performance.emg_prob_estimates = emg_prob_estimates;
% Performance.eeg_decision = eeg_decision;
% % Performance.emg_decision = emg_decision;
% % Performance.comb_decision = comb_decision;
% % Performance.comb_de_thr = comb_de_thr;
% Performance.eeg_CM = eeg_CM;
% % Performance.emg_CM = emg_CM;
% % Performance.comb_CM = comb_CM;
% 
% Performance.eeg_sensitivity = eeg_sensitivity;
% Performance.eeg_specificity = eeg_specificity;
% Performance.eeg_accur = eeg_accur;
% 
% % Performance.emg_sensitivity = emg_sensitivity;
% % Performance.emg_specificity = emg_specificity;
% % Performance.emg_accur = emg_accur;
% % 
% % Performance.comb_sensitivity = comb_sensitivity;
% % Performance.comb_specificity = comb_specificity;
% % Performance.comb_accur = comb_accur;
% 
% % Miscellaneous
% Performance.test_trials_time = test_trials_time;
% Performance.test_trial_signal_time =  test_trial_signal_time;
% 
% file_identifier = [];
% if use_conventional_features == 1
%     file_identifier = [ file_identifier '_fixed'];
% end
% if use_smart_features == 1
%     file_identifier = [ file_identifier '_smart'];
% end
% 
% 
% filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized' file_identifier '.mat']; %datestr(now,'dd_mm_yy_HHMM')
% save(filename2,'Performance');               
%% Extract feature vectors from move and rest epochs -- Time domain features; Not used for classification
% num_feature_per_chan = mlim2 - mlim1 + 1;
% data_set = [];
% %peaks = zeros(2*no_epochs,length(classchannels));
% % AUC = zeros(2*no_epochs,length(classchannels));
% 
% for chan_ind = 1:length(classchannels)
%     move_chn = move_epochs_s(:,:,classchannels(chan_ind));
%     rest_chn = rest_epochs_s(:,:,classchannels(chan_ind));
%     
%     feature_space = [move_chn(:,mlim1:mlim2); rest_chn(:,rlim1:rlim2)];
%     %peaks(:,chan_ind) = min(feature_space,[],2);
%     %AUC(:,chan_ind) = [ trapz(move_erp_time(mlim1:mlim2),move_chn(:,mlim1:mlim2),2); ...
%     %                    trapz(rest_erp_time(rlim1:rlim2),rest_chn(:,rlim1:rlim2),2)];
%     data_set = [data_set feature_space];
% end
% 
% data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];
%%

% conventional features
% if length(window_length_range) > 1
%     for use_feature = 1:2               
%        % Choose optimal_wl
%        for roc_wl = 1:length(window_length_range) 
%            wl_CM = squeeze(All_total_CM{roc_wl});
%            roc_measures(roc_wl,1:2) = [wl_CM(1,1)/(wl_CM(1,1) + wl_CM(1,2)) ...  % TPR
%                                 wl_CM(2,1)/(wl_CM(2,1) + wl_CM(2,2))]; % FPR
%            roc_measures(roc_wl,3) = trace(wl_CM)/sum(sum(wl_CM));   % Accuracy
%            my_prob_est = [];
%            my_true_labels = [];
%            X_Y_Thr = [];
%            for kfold = 1:10
%                each_fold_prob_est = All_eeg_prob_estimates{1,roc_wl}{1,kfold}(:,1);     % Go Probability
%                my_prob_est = [my_prob_est; each_fold_prob_est];
%                my_true_labels = [my_true_labels; [ones(length(each_fold_prob_est)/2,1); 2*ones(length(each_fold_prob_est)/2,1)]];
%            end
%            [X_Y_Thr(:,1),X_Y_Thr(:,2),X_Y_Thr(:,3),roc_OPT_AUC(roc_wl,3),roc_OPT_AUC(roc_wl,1:2)] = ...
%                perfcurve(my_true_labels,my_prob_est,1);
%            roc_X_Y_Thr{roc_wl} = X_Y_Thr;
%        end
%     end

