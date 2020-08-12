function [Averaged_EMG_epochs] = Process_All_EMG_signals(Subject_name, closeloop_Sess_num, loademg_files, blocks_nos_to_load, remove_corrupted_epochs, impaired_hand, process_raw_emg, extract_epochs, session_performance,move_trig_label)
% Function to process bot impaired and unimpaired side EMG
% Created By:  Nikunj Bhagat, PhD
% Contact : nbhagat08[at]gmail.com
% Date Created - 12/24/2019
%% ***********Revisions 


%--------------------------------------------------------------------------------------------------
%% Global variables 

myColors = ['r','b','k','m','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

EMG_channel_nos = [17 22 41 42 45 46 51 55];

% Subject Details   
% Subject_name = 'S9007'; % change1
% closeloop_Sess_num = '3';     % For saving data
% readbv_files = 0;   % Added 8-28-2015
% blocks_nos_to_import = [2 3 4 5];
% process_raw_eeg = 1;         % Also remove extra 'S  2' triggers
% process_raw_emg = 1; 
extract_emg_epochs = 1;
% extract_epochs = 1;     % extract move and rest epochs
% remove_corrupted_epochs = []; %change5
manipulate_epochs = 1;
plot_ERPs = 1;
Block_num = length(blocks_nos_to_load)*20;

closeloop_folder_path = ['D:\NRI_Project_Data\Clinical_study_Data\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3
              
% Pre-processing Flags 
use_noncausal_filter = 0; %always 0; 1 - yes, use zero-phase filtfilt(); 0 - No, use filter()            %change6
use_fir_filter = 0; % always 0
use_band_pass = 0; %always 0, because 4th order bandpass butterworth filter is unstable

%3. Extracting epochs  - only interested in movement epochs, ignore rest
% epcohs
% move_trig_label = 'EMGonset';  % 'S 32'; %'S  8'; %'100';
% rest_trig_label = 'S  2';  % 'S  2'; %'200';
% target_reached_trig_label = 'S  8';
move_epoch_dur = [-3.5 5]; % [-4 12.1];
rest_epoch_dur = [-3.5 5];

%4. Working with Epochs
baseline_int = [-1 -0.5];   
apply_baseline_correction = 0; %Always 0 
All_EMGonset_trial_information = [];

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 

%% %% Load raw EMG datasets to EEGLAB
% if loademg_files == 1
%     total_no_of_trials = 0;    
%     EMG_dataset_to_merge = [];
%     All_EMGonset_trial_information = [];
%     
%     for block_index = 1:length(blocks_nos_to_load)
%         fprintf('\nLoading block # %d of %d blocks...\n',blocks_nos_to_load(block_index),length(blocks_nos_to_load));       
%         total_no_of_trials = total_no_of_trials + 20;
%         EEG = pop_loadset([Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_load(block_index)) '_emg_preprocessed.set'], closeloop_folder_path);
%         
%         % Update EEGLAB window
%         [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%         %    eeglab redraw;
%         EMG_dataset_to_merge = [EMG_dataset_to_merge CURRENTSET];
%         
%         % Preprocess EMG signals
%         if process_raw_emg == 1            
%             emg_Fs = EEG.srate;
%             raw_emg = EEG.data;
%             raw_emg_t = 0:1/emg_Fs: (length(raw_emg) - 1)/emg_Fs;        
% 
%             % ****STEP 2: Bandpass Filter EMG signal, 30 - 200 Hz, 4th order Butterworth
%             BPFred_emg = [];
%             Diff_emg = [];
%             EMG_rms = [];
% 
%             emgbpfc = [30 200];     % Cutoff frequency = 30 - 200 Hz
%             [num_bpf,den_bpf] = butter(4,[emgbpfc(1)/(emg_Fs/2) emgbpfc(2)/(emg_Fs/2)]);
% 
%             if use_noncausal_filter == 1
%                 BPFred_emg = filtfilt(num_bpf,den_bpf,double(raw_emg)')';    
%             else
%                 BPFred_emg = filter(num_bpf,den_bpf,double(raw_emg)')';
%             end
% 
%             Diff_emg(1,:) = BPFred_emg(4,:) - BPFred_emg(3,:);      % Left Biceps
%             Diff_emg(2,:) = BPFred_emg(7,:) - BPFred_emg(1,:);      % Left Triceps
%             Diff_emg(3,:) = BPFred_emg(5,:) - BPFred_emg(6,:);      % Right Biceps
%             Diff_emg(4,:) = BPFred_emg(8,:) - BPFred_emg(2,:);      % Right Triceps   
%             emgt = 0:1/emg_Fs:(size(EEG.data,2)-1)/emg_Fs;
%         %     figure; 
%         %     subplot(2,1,1); 
%         %     plot(emgt, Diff_emg(1,:),'b');
%         %     hold on; plot(emgt, Diff_emg(2,:),'r');
%         %     legend('L.biceps', 'L.triceps');
%         %     title('Left hand');
%         %     
%         %     subplot(2,1,2); 
%         %     plot(emgt, Diff_emg(3,:),'k');
%         %     hold on; plot(emgt, Diff_emg(4,:),'g');
%         %     legend('R.biceps', 'R.triceps');
%         %     title('Right hand');
%         %     
% 
%         %     % Apply notch filter
%         %     [z p k] = butter(4, [4 7]./(emg_Fs/2), 'stop'); % 10th order filter
%         %     [sos,g]=zp2sos(z,p,k); % Convert to 2nd order sections form
%         %     notchf=dfilt.df2sos(sos,g); % Create filter object
%         %     %fvtool(notchf,'Analysis','freq','Fs',emg_Fs);
%         %     for i = 1:size(raw_emg,1)
%         %         raw_emg(i,:) = filtfilt(notchf.sos.sosMatrix,notchf.sos.ScaleValues,raw_emg(i,:));
%         %     end
% 
%             % ****STEP 2: Rectify (Instead use TKEO) & Lowpass Filter EMG Envelop, 1 Hz, 4th order Butterworth
% 
%             %**** STEP3: Calculate RMS using 300ms sliding window
%            EMG_rms(1,:) = sqrt(smooth(Diff_emg(1,:).^2,150));
%            EMG_rms(2,:) = sqrt(smooth(Diff_emg(2,:).^2,150));
%            EMG_rms(3,:) = sqrt(smooth(Diff_emg(3,:).^2,150));
%            EMG_rms(4,:) = sqrt(smooth(Diff_emg(4,:).^2,150));
% 
% %            figure; 
% %            subplot(2,1,1); 
% %            plot(emgt, EMG_rms(1,:),'b');
% %            hold on; plot(emgt, EMG_rms(2,:),'r');
% %            legend('L.biceps', 'L.triceps');
% %            title('Left hand (RMS)');
% %            ylim([0 50]);
% %         
% %            subplot(2,1,2); 
% %            plot(emgt, EMG_rms(3,:),'k');
% %            hold on; plot(emgt, EMG_rms(4,:),'g');
% %            legend('R.biceps', 'R.triceps');
% %            title('Right hand (RMS)');
% %            ylim([0 50]);
% 
%            %**** STEP4: Apply TK Energy operator to Diff_emg
%            TKEO_emg = applyTKEO(Diff_emg);
%            emglpfc = 2;       
%            [num_lpf,den_lpf] = butter(8,(emglpfc/(emg_Fs/2)));
%         %    fvtool(num_lpf, den_lpf)
%            if use_noncausal_filter == 1                
%                TKEO_emgf = filtfilt(num_lpf,den_lpf,TKEO_emg')';
%            else   
%                TKEO_emgf = filter(num_lpf,den_lpf,TKEO_emg')';
%         %        TKEO_emgf = envelope(TKEO_emg',300,'peak')';
%            end       
%            
%            %**** STEP5: Detect EMG onset 
%            % A. Zero first 30 seconds of TKEO
%            TKEO_emgf(:,1:30*emg_Fs) = 0;
% 
%            % Removed this step, because we will directly import EMGonset markers from previous analysis - Nikunj 12/24/2019
% %            % B. Zero impossibly high values
% %            for i = 1:4       
% %                ridiculously_high_value = find(TKEO_emgf(i,:) > 1E4);
% %                for j = 1:length(ridiculously_high_value)
% %                    if ridiculously_high_value(j) >= 250         %0.5sec
% %                         TKEO_emgf(i,ridiculously_high_value(j)-250:ridiculously_high_value(j)) = 0;                
% %                    end
% %                end             
% %            end
% 
%            %C. Detrend the data
%            TKEO_emgf = detrend(TKEO_emgf','constant')';
%            
%            %D. Calculate mean and s.d of TKEO after ignoring first 30seconds of
%            % data
% %            mTKEO_emgf = mean(TKEO_emgf(:,30*emg_Fs+1:end),2); 
% %            stdTKEO_emgf = std(TKEO_emgf(:,30*emg_Fs+1:end),0,2); 
% 
% %            figure; 
% %            subplot(2,1,1); 
% %            plot(emgt, TKEO_emgf(1,:),'b');
% %            hold on; plot(emgt, TKEO_emgf(2,:),'r');
% %            legend('L.biceps', 'L.triceps');
% %            title('Left hand (TKEO)');
% %            ylim([0 5000]);
% % 
% %            subplot(2,1,2); 
% %            plot(emgt, TKEO_emgf(3,:),'k');
% %            hold on; plot(emgt, TKEO_emgf(4,:),'g');
% %            legend('R.biceps', 'R.triceps');
% %            title('Right hand (TKEO)');
% %            ylim([0 5000]); 
% 
% % Commented out by Nikunj on 12/24/2019 because we will no longer detect
% % EMGonset
% 
% % %            if strcmp(impaired_hand,'L')
% % %                % Detect onset using left hand ativity
% % %                FlexOnset = (TKEO_emgf(1,:) >= stdTKEO_emgf(1));
% % %                ExtOnset = (TKEO_emgf(2,:) >= stdTKEO_emgf(2));
% % % 
% % %            elseif strcmp(impaired_hand,'R')
% % %                % otherwise right hand 
% % %                FlexOnset = (TKEO_emgf(3,:) >= stdTKEO_emgf(3));
% % %                ExtOnset = (TKEO_emgf(4,:) >= stdTKEO_emgf(4));
% % %            end
% % %            CombinedOnset = FlexOnset | ExtOnset;
% % %            [Movement_Onset, CombinedOnsetUnique] = ExtractUniqueTriggers(CombinedOnset, 1, 1);
% % % 
% % %            [Movement_Offset, CombinedOffsetUnique] = ExtractUniqueTriggers(5.*[1, CombinedOnset], 1, 0); % Multiply by 5 to be consistent with InMotion
% % % 
% % %            %activeEMG_duration = [Movement_Offset(2:end);length(emgt)] - Movement_Onset;   
% % %            activeEMG_duration = [Movement_Offset(2:length(Movement_Onset));length(emgt)] - Movement_Onset;   
% % %            %figure; hist(activeEMG_duration./500,100)
% % % 
% % %            EMG_onset_detected = (Movement_Onset(activeEMG_duration./emg_Fs >= 1))./emg_Fs;    
% % % 
% % %             % **** STEP 6: Keep only the EMG onset detected events that occur
% % %             % during a trial, i.e. between 'S  2' and 'S  16' or 'S  1'    
% % %             % Note: 'S  1' is missing in some sessions, hence use Time_to_Trigger
% % %             % instead
% % %             start_of_trial_events = find(strcmp({EEG.event.type},'S  2')); %| strcmp({EEG.event.type},'S 18'));    
% % %             %end_of_trial_events = find(strcmp({EEG.event.type},'S 16') | strcmp({EEG.event.type},'S  1'));
% % % 
% % %             valid_EMG_onset_detected = []; 
% % %             start_of_trial_timestamps = zeros(length(start_of_trial_events),1);
% % %             end_of_trial_timestamps = zeros(length(start_of_trial_events),1);
% % %             Time_to_Trigger_blockwise = Time_to_Trigger_sessionwise(Time_to_Trigger_sessionwise(:,1) == blocks_nos_to_load(block_index),2);
% % %             for SOTindex = 1:length(start_of_trial_events)
% % %                 start_of_trial_timestamps(SOTindex) = EEG.event(start_of_trial_events(SOTindex)).latency/emg_Fs;        
% % %                 end_of_trial_timestamps(SOTindex) = start_of_trial_timestamps(SOTindex) + Time_to_Trigger_blockwise(SOTindex) + 3;        % wait additional 3sec
% % % 
% % %                 valid_EMG_onset_detected = [valid_EMG_onset_detected;...
% % %                     EMG_onset_detected((EMG_onset_detected >= start_of_trial_timestamps(SOTindex)) & ...
% % %                     (EMG_onset_detected <= end_of_trial_timestamps(SOTindex)))];                
% % %             end
% % % 
% % %             figure; hold on;
% % %             title([Subject_name, 'Session #' num2str(closeloop_Sess_num)...
% % %                 ', block# ' num2str(blocks_nos_to_load(block_index))]);
% % %             if strcmp(impaired_hand,'L')
% % %                plot(emgt, TKEO_emgf(1:2,:));          
% % %             elseif strcmp(impaired_hand,'R')
% % %                plot(emgt, TKEO_emgf(3:4,:));          
% % %             end    
% % %         %     for i = 1:length(EMG_onset_detected)
% % %         %        line([EMG_onset_detected(i) EMG_onset_detected(i)], ...
% % %         %            [0 5000], 'Color', 'r');
% % %         %     end
% % %             for i = 1:length(valid_EMG_onset_detected)
% % %                line([valid_EMG_onset_detected(i) valid_EMG_onset_detected(i)], ...
% % %                    [0 5000], 'Color', 'k');
% % %             end
% % %         %     for i = 1:length(start_of_trial_timestamps)
% % %         %         line([start_of_trial_timestamps(i) start_of_trial_timestamps(i)], ...
% % %         %            [0 6000], 'Color', 'g');
% % %         %        line([end_of_trial_timestamps(i) end_of_trial_timestamps(i)], ...
% % %         %            [0 6000], 'Color', 'm');
% % %         %     end
% % %         %     ylim([0 5E3])
% 
%             EEG.data(1,:) = EMG_rms(1,:);
%             EEG.data(2,:) = EMG_rms(2,:);
%             EEG.data(3,:) = EMG_rms(3,:);
%             EEG.data(4,:) = EMG_rms(4,:);
%             EEG.data(5,:) = TKEO_emgf(1,:);
%             EEG.data(6,:) = TKEO_emgf(2,:);
%             EEG.data(7,:) = TKEO_emgf(3,:);
%             EEG.data(8,:) = TKEO_emgf(4,:);
%             
%             marker_filename = [closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
%                     '_closeloop_block' num2str(blocks_nos_to_load(block_index)) '_EMGonset_markers.txt'];
%             if (~exist(marker_filename,'file'))                            
%                 warndlg(['No EMG onset detected in Session', num2str(closeloop_Sess_num), ', Block ' num2str(blocks_nos_to_load(block_index))]);
%             else
%                 EEG = pop_importevent( EEG, 'event',marker_filename,'fields',{'type' 'latency'},'timeunit',1,'align',0);
%                 EEG = eeg_checkset( EEG );
%                 EMGonset_markers = dlmread(marker_filename,' ',0,2);                 
%             end
%                         
%             % Update EEGLAB window
%             [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%             EMG_dataset_to_merge(block_index) = CURRENTSET;
% %             eeglab redraw;    
%             EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) ...
%                  '_closeloop_block' num2str(blocks_nos_to_load(block_index)) '_emg_preprocessed.set'],'filepath',closeloop_folder_path);        
%              
%             % Find out whether flexion or extension occurs first
%             block_performance = session_performance(session_performance(:,2) == blocks_nos_to_load(block_index),:); 
%             eeg_kin_time_difference = EEG.event(strcmp({EEG.event.type},'S 42')).latency/EEG.srate;             
%             start_of_trial_times_from_kinematics = block_performance(:,3)/1000 + eeg_kin_time_difference;
%             end_of_trial_times_from_kinematics = block_performance(:,4)/1000 + eeg_kin_time_difference;            
%             [nr,nc] = size(EMGonset_markers);
%             if  nc > 1
%                EMGonset_markers = EMGonset_markers(:,1);
%             end
%             EMGonset_trial_information = zeros(nr,2); % Col 1 - trial no, Col 2 - Intended target (i.e. 1 - flexion, 3 - extension)
%             for i = 1:nr
%                 for j = 1:length(start_of_trial_times_from_kinematics)
%                     if (EMGonset_markers(i) >= start_of_trial_times_from_kinematics(j)) && (EMGonset_markers(i) <= end_of_trial_times_from_kinematics(j))
%                        EMGonset_trial_information(i,:) = [j, block_performance(j,19)];
%                     end
%                 end
%             end
%             
%         end      
%         All_EMGonset_trial_information = [All_EMGonset_trial_information, EMGonset_trial_information];
%     end
%             
%     if ~isempty(EMG_dataset_to_merge)
%         fprintf('All block have been loaded. Merging %d blocks...\n', length(EMG_dataset_to_merge));
%         if length(EMG_dataset_to_merge) == 1
%             % Retrive old data set - No need to merge 
%             [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EMG_dataset_to_merge,'study',0); 
%         else
%             EEG = pop_mergeset( ALLEEG,EMG_dataset_to_merge, 0);
%         end
%         EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_emg_preprocessed'];
%         EEG = eeg_checkset( EEG );
%         EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_emg_preprocessed.set'],'filepath',closeloop_folder_path);
%         EEG = eeg_checkset( EEG );
%     else
%         errordlg('Error: EMG blocks could not be merged');
%     end
%         
%     % Update EEGLAB window
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%     eeglab redraw;
% end
%% Extract move epochs and rest epochs
if extract_epochs == 1
    
    %% Extract move epochs
    EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_emg_preprocessed.set'], closeloop_folder_path);    
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    %eeglab redraw;        
       
    EEG = pop_epoch( EEG, {  move_trig_label  }, move_epoch_dur, 'newname', 'EMG_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    
    if ~isempty(remove_corrupted_epochs)
        reject_epochs = zeros(1,size(EEG.epoch,2));
        reject_epochs(remove_corrupted_epochs) = 1;
        EEG = pop_rejepoch(EEG,reject_epochs,1);
    else
%         pop_eegplot(pop_select(EEG,'channel',Channels_nos(6:end)),1,1,0)
    end

    EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_emg_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_emg_epochs.set'],'filepath',closeloop_folder_path);    
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end

%% Manipulate & Calculate ERPs for Move Epochs
if manipulate_epochs == 1            
    EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_emg_epochs.set'], closeloop_folder_path);
    Fs_eeg = EEG.srate;
    move_erp_time = round((EEG.xmin:1/Fs_eeg:EEG.xmax)*1000)./1000;

    [no_channels,no_datapts,no_epochs] = size(EEG.data);
    move_epochs = zeros([no_epochs,no_datapts,no_channels]); 
    move_avg_channels = zeros(no_channels,no_datapts);
    move_std_channels = zeros(no_channels,no_datapts);
    move_SE_channels = [];
    
    % Determine t-statistics for 95% C.I.
    %http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s    
    deg_freedom = no_epochs - 1;
    t_value = tinv(1 - 0.05/2, deg_freedom);

    for epoch_cnt = 1:no_epochs
        for channel_cnt = 1:no_channels
            move_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt);
            move_mean_baseline(channel_cnt,epoch_cnt) = mean(move_epochs(epoch_cnt,find(move_erp_time == (baseline_int(1))):find(move_erp_time == (baseline_int(2))),channel_cnt));            
            move_epochs_with_base_correct(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt) - move_mean_baseline(channel_cnt,epoch_cnt);            
        end
    end

    

    % Determine t-statistics for 95% C.I.
    %http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s    
    deg_freedom = size(move_epochs_with_base_correct,1) - 1;
    t_value = tinv(1 - 0.05/2, deg_freedom);
    
    for channel_cnt = 1:no_channels    
        move_avg_channels(channel_cnt,:) =  mean(move_epochs_with_base_correct(:,:,channel_cnt));
        move_std_channels(channel_cnt,:) =  std(move_epochs_with_base_correct(:,:,channel_cnt));
        move_SE_channels(channel_cnt,:) = t_value.*std(move_epochs_with_base_correct(:,:,channel_cnt))/sqrt(size(move_epochs_with_base_correct,1));
    end
end
%% Save data
if manipulate_epochs == 1
   Averaged_EMG_epochs.move_epochs = move_epochs;
   Averaged_EMG_epochs.move_epochs_with_base_correct = move_epochs_with_base_correct;
%    Posthoc_Average.rest_epochs = rest_epochs;
   Averaged_EMG_epochs.Fs_eeg = Fs_eeg;
   Averaged_EMG_epochs.apply_baseline_correction = apply_baseline_correction;
   Averaged_EMG_epochs.All_EMGonset_trial_information = All_EMGonset_trial_information;

   Averaged_EMG_epochs.move_avg_channels = move_avg_channels;
   Averaged_EMG_epochs.move_std_channels = move_std_channels;
   Averaged_EMG_epochs.move_SE_channels = move_SE_channels;
   Averaged_EMG_epochs.move_mean_baseline = move_mean_baseline;   
   Averaged_EMG_epochs.move_erp_time = move_erp_time;
%    Averaged_EMG_epochs.epochs_with_artefacts = epochs_with_artefacts;

%    Average.rest_avg_channels = rest_avg_channels;
%    Average.rest_std_channels = rest_std_channels;
%    Average.rest_SE_channels = rest_SE_channels;
%    Average.rest_mean_baseline = rest_mean_baseline;   
%    Average.rest_erp_time = rest_erp_time;
   Averaged_EMG_epochs.remove_corrupted_epochs = remove_corrupted_epochs;
   
%    disp('Suggested RP channels = '); disp(Channels_criteria1_and_2);
%    Average.RP_chans = input('Enter channel numbers where RP is seen. Example - [1 2 3 4]: ');
%    Averaged_EMG_epochs.RP_chans = Performance.classchannels;
%    Averaged_EMG_epochs.optimized_channels = Performance.optimized_channels;   
%    Averaged_EMG_epochs.bad_move_trials = bad_move_trials;
%    Averaged_EMG_epochs.good_move_trials = good_move_trials;
%    Averaged_EMG_epochs.smart_window_length = Performance.smart_window_length;   
%    Averaged_EMG_epochs.Smart_Features = Smart_Features;       % New Features
   
   filename1 = [closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_averaged_emg_epochs.mat'];
   save(filename1,'Averaged_EMG_epochs');   
end