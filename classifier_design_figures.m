% Plot figures for classifier design steps
clear;

% Subject Details
Subject_name = 'BNBO';
Sess_num = '2';
Cond_num = 1;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;

folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_average_causal.mat']);      % Always use causal for training classifier
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier

Fs_eeg = 20; % Desired sampling frequency in real-time
downsamp_factor = Average.Fs_eeg/Fs_eeg;
move_epochs = Average.move_epochs;
rest_epochs = Average.rest_epochs;
[no_epochs,no_datapts,no_channels] = size(Average.move_epochs); %#ok<*ASGLU>

%-. Use separate move and rest epochs? (Always yes). 0 - No; 1 - yes
use_separate_move_rest_epochs = 1; 

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

    classchannels = Performance.classchannels;
   % Find peak within interval, Reject trials reject_trial_onwards 
    find_peak_interval = [-2.0 0.5];            % ranges up to 0.5 sec after movement onset because of phase delay filtering
    reject_trial_before = -1.5; % Seconds. reject trials for which negative peak is reached earlier than -1.5 sec
    
%% Plot ERP image plot for classifier channels after sorting
bad_move_trials = []; 
good_move_trials = [];
move_ch_avg_ini = mean(move_epochs_s(:,:,classchannels),3);
rest_ch_avg_ini = mean(rest_epochs_s(:,:,classchannels),3);

mt1 = find(move_erp_time == find_peak_interval(1));
mt2 = find(move_erp_time == find_peak_interval(2));
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

%% Plot sensitivity & specificity
% figure; 
% group_names = {'Online Fixed','Online Flexible'};
% online_fixed = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);
% online_variable = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
% 
% % filename3 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_smart_offline_nic.mat'];
% % load(filename3);
% % offline_fixed = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);
% 
% 
% h = boxplot([online_fixed' online_variable'] ,'labels',group_names,'widths',0.5);
%  set(h,'LineWidth',2);
% v = axis;
% axis([v(1) v(2) 0 100]);
% ylabel('Classification Accuracy (%)','FontSize',12);
% title([Subject_name ', Mode ' num2str(Cond_num)],'FontSize',12);
% 
% %export_fig 'TA_ses1_cond3_smart_conv_comparison' '-png' '-transparent'
% 
% % if use_shifting_window_CV == 1
% %     title('Online Simulation','FontSize',12);
% %     %mtit('Online Simulation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
% % else
% %     title('Offline Validation','FontSize',12);
% %     %mtit('Offline Validation','fontsize',12,'color',[0 0 1],'xoff',0,'yoff',-1.15);
% % end
