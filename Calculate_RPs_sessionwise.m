function [Posthoc_Average] = Calculate_RPs_sessionwise(Subject_name, closeloop_Sess_num, readbv_files, blocks_nos_to_import, remove_corrupted_epochs, impaired_hand, Addl_session_info, Performance, process_raw_emg, process_raw_eeg, extract_epochs)

% Function for extracting RPs from several blocks of closed-loop BMI data in a session  
% Created By:  Nikunj Bhagat, PhD
% Contact : nbhagat08[at]gmail.com
% Date Created - 7/6/2019
%% ***********Revisions 


%--------------------------------------------------------------------------------------------------
%% Global variables 

myColors = ['r','b','k','m','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% EEG Channels used for identifying MRCP
Channels_nos = [ 37,   4, 38,   5,  39,   6,  40,... 
                  8, 43,   9, 32, 10, 44,  11,...
                 47, 13, 48, 14, 49, 15, 50,...
                 18, 52, 19, 53, 20, 54, 21,... 
                 56, 24, 57, 25, 58, 26, 59];    % 32 or 65 for FCz

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
Block_num = length(blocks_nos_to_import)*20;

closeloop_folder_path = ['D:\NRI_Project_Data\Clinical_study_Data\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3
              
% Pre-processing Flags
%1. Spatial Filter Type
spatial_filter_type = 'LLAP';
    
%2. Filter cutoff frequency  
hpfc = 0.1;     % HPF Cutoff frequency = 0.1 Hz    
lpfc = 1;      % LPF Cutoff frequency = 1 Hz
use_noncausal_filter = 0; %always 0; 1 - yes, use zero-phase filtfilt(); 0 - No, use filter()            %change6
use_fir_filter = 0; % always 0
use_band_pass = 0; %always 0, because 4th order bandpass butterworth filter is unstable

%3. Extracting epochs  - only interested in movement epochs, ignore rest
% epcohs
move_trig_label = 'EMGonset';  % 'S 32'; %'S  8'; %'100';
rest_trig_label = 'S  2';  % 'S  2'; %'200';
target_reached_trig_label = 'S  8';
move_epoch_dur = [-3.5 2]; % [-4 12.1];
rest_epoch_dur = [-3.5 2];

%4. Working with Epochs
baseline_int = [-2.5 -2.25];   
apply_baseline_correction = 0;  % Always

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 
%% Import raw BrainVision files (.eeg, .vhdr, .vmrk)  to EEGLAB dataset
if readbv_files == 1
    total_no_of_trials = 0;
    EEG_dataset_to_merge = [];
    EMG_dataset_to_merge = [];
              
    for block_index = 1:length(blocks_nos_to_import)
        fprintf('\nImporting block # %d of %d blocks...\n',blocks_nos_to_import(block_index),length(blocks_nos_to_import));       
        total_no_of_trials = total_no_of_trials + 20;
        if blocks_nos_to_import(block_index) > 9
             EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block00' num2str(blocks_nos_to_import(block_index)) '.vhdr'], [], 1:64);
        else           
             EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block000' num2str(blocks_nos_to_import(block_index)) '.vhdr'], [], 1:64);           
        end

        EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_eeg_raw'];
        EEG = eeg_checkset( EEG );
        
        EEG = pop_chanedit(EEG, 'lookup','D:\NRI_Project_Data\eeglab2019_0\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp');
        EEG = eeg_checkset( EEG );
%        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_eeg_raw.set'],...
%            'filepath',closeloop_folder_path);
%        EEG = eeg_checkset( EEG );
        % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        %    eeglab redraw;
        EEG_dataset_to_merge = [EEG_dataset_to_merge CURRENTSET];

        EEG = pop_select( EEG,'channel',EMG_channel_nos);
        EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_emg_raw'];
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_emg_raw.set'],...
            'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
        % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        %    eeglab redraw;        
        EMG_dataset_to_merge = [EMG_dataset_to_merge CURRENTSET];
        % Preprocess EMG signals
        if process_raw_emg == 1            
            emg_Fs = EEG.srate;
            raw_emg = EEG.data;
            raw_emg_t = 0:1/emg_Fs: (length(raw_emg) - 1)/emg_Fs;        

            % ****STEP 2: Bandpass Filter EMG signal, 30 - 200 Hz, 4th order Butterworth
            BPFred_emg = [];
            Diff_emg = [];
            EMG_rms = [];

            emgbpfc = [30 200];     % Cutoff frequency = 30 - 200 Hz
            [num_bpf,den_bpf] = butter(4,[emgbpfc(1)/(emg_Fs/2) emgbpfc(2)/(emg_Fs/2)]);

            if use_noncausal_filter == 1
                BPFred_emg = filtfilt(num_bpf,den_bpf,double(raw_emg)')';    
            else
                BPFred_emg = filter(num_bpf,den_bpf,double(raw_emg)')';
            end

            Diff_emg(1,:) = BPFred_emg(4,:) - BPFred_emg(3,:);      % Left Biceps
            Diff_emg(2,:) = BPFred_emg(7,:) - BPFred_emg(1,:);      % Left Triceps
            Diff_emg(3,:) = BPFred_emg(5,:) - BPFred_emg(6,:);      % Right Biceps
            Diff_emg(4,:) = BPFred_emg(8,:) - BPFred_emg(2,:);      % Right Triceps   
            emgt = 0:1/emg_Fs:(size(EEG.data,2)-1)/emg_Fs;
        %     figure; 
        %     subplot(2,1,1); 
        %     plot(emgt, Diff_emg(1,:),'b');
        %     hold on; plot(emgt, Diff_emg(2,:),'r');
        %     legend('L.biceps', 'L.triceps');
        %     title('Left hand');
        %     
        %     subplot(2,1,2); 
        %     plot(emgt, Diff_emg(3,:),'k');
        %     hold on; plot(emgt, Diff_emg(4,:),'g');
        %     legend('R.biceps', 'R.triceps');
        %     title('Right hand');
        %     

        %     % Apply notch filter
        %     [z p k] = butter(4, [4 7]./(emg_Fs/2), 'stop'); % 10th order filter
        %     [sos,g]=zp2sos(z,p,k); % Convert to 2nd order sections form
        %     notchf=dfilt.df2sos(sos,g); % Create filter object
        %     %fvtool(notchf,'Analysis','freq','Fs',emg_Fs);
        %     for i = 1:size(raw_emg,1)
        %         raw_emg(i,:) = filtfilt(notchf.sos.sosMatrix,notchf.sos.ScaleValues,raw_emg(i,:));
        %     end

            % ****STEP 2: Rectify (Instead use TKEO) & Lowpass Filter EMG Envelop, 1 Hz, 4th order Butterworth

            %**** STEP3: Calculate RMS using 300ms sliding window
           EMG_rms(1,:) = sqrt(smooth(Diff_emg(1,:).^2,150));
           EMG_rms(2,:) = sqrt(smooth(Diff_emg(2,:).^2,150));
           EMG_rms(3,:) = sqrt(smooth(Diff_emg(3,:).^2,150));
           EMG_rms(4,:) = sqrt(smooth(Diff_emg(4,:).^2,150));

%            figure; 
%            subplot(2,1,1); 
%            plot(emgt, EMG_rms(1,:),'b');
%            hold on; plot(emgt, EMG_rms(2,:),'r');
%            legend('L.biceps', 'L.triceps');
%            title('Left hand (RMS)');
%            ylim([0 50]);
%         
%            subplot(2,1,2); 
%            plot(emgt, EMG_rms(3,:),'k');
%            hold on; plot(emgt, EMG_rms(4,:),'g');
%            legend('R.biceps', 'R.triceps');
%            title('Right hand (RMS)');
%            ylim([0 50]);

           %**** STEP4: Apply TK Energy operator to Diff_emg
           TKEO_emg = applyTKEO(Diff_emg);
           emglpfc = 0.5;       
           [num_lpf,den_lpf] = butter(4,(emglpfc/(emg_Fs/2)));
%            fvtool(num_lpf, den_lpf)
           if use_noncausal_filter == 1                
               TKEO_emgf = filtfilt(num_lpf,den_lpf,TKEO_emg')';
           else   
               TKEO_emgf = filter(num_lpf,den_lpf,TKEO_emg')';
        %        TKEO_emgf = envelope(TKEO_emg',300,'peak')';
           end       
           
           %**** STEP5: Detect EMG onset 
           % A. Zero first 30 seconds of TKEO
           TKEO_emgf(:,1:30*emg_Fs) = 0;

           % B. Zero impossibly high values
%            for i = 1:4       
%                ridiculously_high_value = find(TKEO_emgf(i,:) > 1E5); % Changed from 1E4 on 12-25-2019
% %                for j = 1:length(ridiculously_high_value)
% %                    if ridiculously_high_value(j) >= 250         %0.5sec
% %                         TKEO_emgf(i,ridiculously_high_value(j)-250:ridiculously_high_value(j)) = 0;                
% %                    end
% %                end             
%            end

           %C. Detrend the data
%            TKEO_emgf = detrend(TKEO_emgf','constant')';
           
           %D. Calculate mean and s.d of TKEO after ignoring first 30seconds of
           % data
%            mTKEO_emgf = mean(TKEO_emgf(:,30*emg_Fs+1:end),2); 
%            stdTKEO_emgf = std(TKEO_emgf(:,30*emg_Fs+1:end),0,2); 

           %B. Directly zscore the data
           TKEO_emgf_original = TKEO_emgf;
           TKEO_emgf(:,30*emg_Fs+1:end) = zscore(TKEO_emgf(:,30*emg_Fs+1:end)')';
           
%            figure; 
%            subplot(2,1,1); 
%            plot(emgt, TKEO_emgf(1,:),'b');
%            hold on; plot(emgt, TKEO_emgf(2,:),'r');
%            legend('L.biceps', 'L.triceps');
%            title('Left hand (TKEO)');
%            ylim([0 5000]);
% 
%            subplot(2,1,2); 
%            plot(emgt, TKEO_emgf(3,:),'k');
%            hold on; plot(emgt, TKEO_emgf(4,:),'g');
%            legend('R.biceps', 'R.triceps');
%            title('Right hand (TKEO)');
%            ylim([0 5000]); 

           if strcmp(impaired_hand,'L')
               % Detect onset using left hand ativity
               FlexOnset = (TKEO_emgf(1,:) >= 0.5); % half standard deviation
               ExtOnset = (TKEO_emgf(2,:) >= 0.5);

           elseif strcmp(impaired_hand,'R')
               % otherwise right hand 
               FlexOnset = (TKEO_emgf(3,:) >= 0.5);
               ExtOnset = (TKEO_emgf(4,:) >= 0.5);
           end
           CombinedOnset = FlexOnset | ExtOnset;
           [Movement_Onset, CombinedOnsetUnique] = ExtractUniqueTriggers(CombinedOnset, 1, 1);

           [Movement_Offset, CombinedOffsetUnique] = ExtractUniqueTriggers(5.*[1, CombinedOnset], 1, 0); % Multiply by 5 to be consistent with InMotion

           %activeEMG_duration = [Movement_Offset(2:end);length(emgt)] - Movement_Onset;   
           activeEMG_duration = [Movement_Offset(2:length(Movement_Onset));length(emgt)] - Movement_Onset;   
           %figure; hist(activeEMG_duration./500,100)
           
           EMG_onset_detected_indices = Movement_Onset(activeEMG_duration./emg_Fs >= 1);
%            if strcmp(impaired_hand,'L')
%                % Detect onset using left hand ativity
%                EMGcombination = max(TKEO_emgf([1,2],:));
%                QuietEMGthreshold = max(stdTKEO_emgf(1:2));
%            elseif strcmp(impaired_hand,'R')
%                % otherwise right hand 
%                EMGcombination = max(TKEO_emgf([3,4],:));
%                QuietEMGthreshold = max(stdTKEO_emgf(3:4));
%            end
%            rejected_EMG_onset_detected_indices = [];
%            if (length(EMG_onset_detected_indices) > 1)
%                for index=2:length(EMG_onset_detected_indices)
%                    if ~isempty(find(EMGcombination(EMG_onset_detected_indices(index-1):EMG_onset_detected_indices(index)) <= QuietEMGthreshold))
%                       %do nothing 
%                    else
%                       % remove this index
%                       rejected_EMG_onset_detected_indices = [rejected_EMG_onset_detected_indices; EMG_onset_detected_indices(index)];
%                       EMG_onset_detected_indices(index) = [];
%                    end
%                end
%            end
           EMG_onset_detected = (EMG_onset_detected_indices)./emg_Fs;
%            EMG_onset_rejected = (rejected_EMG_onset_detected_indices)./emg_Fs;

            % **** STEP 6: Keep only the EMG onset detected events that occur
            % during a trial, i.e. between 'S  2' and 'S  16' or 'S  1'    
            % Note: 'S  1' is missing in some sessions, hence use Time_to_Trigger
            % instead
            start_of_trial_events = find(strcmp({EEG.event.type},'S  2')); %| strcmp({EEG.event.type},'S 18'));    
            %end_of_trial_events = find(strcmp({EEG.event.type},'S 16') | strcmp({EEG.event.type},'S  1'));

            valid_EMG_onset_detected = []; 
            start_of_trial_timestamps = zeros(length(start_of_trial_events),1);
            end_of_trial_timestamps = zeros(length(start_of_trial_events),1);
            Time_to_Trigger_blockwise = Addl_session_info(Addl_session_info(:,1) == blocks_nos_to_import(block_index),2);
            Target_movement_blockwise = Addl_session_info(Addl_session_info(:,1) == blocks_nos_to_import(block_index),3);
            for SOTindex = 1:length(start_of_trial_events)
                start_of_trial_timestamps(SOTindex) = EEG.event(start_of_trial_events(SOTindex)).latency/emg_Fs;        
                end_of_trial_timestamps(SOTindex) = min(start_of_trial_timestamps(SOTindex) + Time_to_Trigger_blockwise(SOTindex) + 3,... % wait additional 3sec
                   start_of_trial_timestamps(SOTindex) + 15.001);      % Added min() condition on 12-25-2019
                EMG_onset_indexwise = EMG_onset_detected((EMG_onset_detected >= start_of_trial_timestamps(SOTindex)) & ...
                    (EMG_onset_detected <= end_of_trial_timestamps(SOTindex))); 
                if (~isempty(EMG_onset_indexwise))
                    valid_EMG_onset_detected = [valid_EMG_onset_detected;...
                        [EMG_onset_indexwise, Target_movement_blockwise(SOTindex)*ones(length(EMG_onset_indexwise),1)]...
                    ];                                    
                end                
            end

            figure; hold on;
            title([Subject_name, ', Session #' num2str(closeloop_Sess_num)...
                ', block# ' num2str(blocks_nos_to_import(block_index))]);
            if strcmp(impaired_hand,'L')
               plot(emgt, TKEO_emgf(1:2,:));          
            elseif strcmp(impaired_hand,'R')
               plot(emgt, TKEO_emgf(3:4,:));          
            end    
        %     for i = 1:length(EMG_onset_detected)
        %        line([EMG_onset_detected(i) EMG_onset_detected(i)], ...
        %            [0 5000], 'Color', 'r');
        %     end
            for i = 1:length(valid_EMG_onset_detected)
               line([valid_EMG_onset_detected(i) valid_EMG_onset_detected(i)], ...
                   [0 10], 'Color', 'k');
            end
%             for i = 1:length(EMG_onset_rejected)
%                line([EMG_onset_rejected(i) EMG_onset_rejected(i)], ...
%                    [0 35000], 'Color', [1 0.3 0]);
%             end
            for i = 1:length(start_of_trial_timestamps)
                line([start_of_trial_timestamps(i) start_of_trial_timestamps(i)], ...
                   [0 10], 'Color', 'g');
               line([end_of_trial_timestamps(i) end_of_trial_timestamps(i)], ...
                   [0 10], 'Color', 'm');
            end
            ylim([0 (max(TKEO_emgf(:))+3)]);

            EEG.data(1,:) = EMG_rms(1,:);
            EEG.data(2,:) = EMG_rms(2,:);
            EEG.data(3,:) = EMG_rms(3,:);
            EEG.data(4,:) = EMG_rms(4,:);
            EEG.data(5,:) = TKEO_emgf_original(1,:);
            EEG.data(6,:) = TKEO_emgf_original(2,:);
            EEG.data(7,:) = TKEO_emgf_original(3,:);
            EEG.data(8,:) = TKEO_emgf_original(4,:);
            
            if (~isempty(valid_EMG_onset_detected))
                marker_file_id = fopen([closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
                    '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_EMGonset_markers.txt'],'w');
                for k= 1:size(valid_EMG_onset_detected,1)
                    fprintf(marker_file_id,'EMGonset \t %d \n', valid_EMG_onset_detected(k,1));
                    fprintf(marker_file_id,'Target%d \t %d \n', valid_EMG_onset_detected(k,2),valid_EMG_onset_detected(k,1));
                end
                fclose(marker_file_id);
                EEG = pop_importevent( EEG, 'event',[closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
                    '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_EMGonset_markers.txt'],'fields',{'type' 'latency'},'timeunit',1,'align',0);
                EEG = eeg_checkset( EEG );
            else
                warndlg(['No EMG onset detected in Session', num2str(closeloop_Sess_num), ', Block ' num2str(blocks_nos_to_import(block_index))]);
            end
                        
            % Update EEGLAB window
            [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
            EMG_dataset_to_merge(block_index) = CURRENTSET;
%             eeglab redraw;    
            EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) ...
                 '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_emg_preprocessed.set'],'filepath',closeloop_folder_path);            
        end
        %EEG.data(:,1:(EEG.srate*5)) = repmat(EEG.data(:,EEG.srate*5+1),1,EEG.srate*5);            % Replicate first and last 5 seconds of avoid discontinuity or sudden jumps in data, which fails the EMG onset detection algorithm
        %EEG.data(:,end-(EEG.srate*5 - 1):end) = repmat(EEG.data(:,end - EEG.srate*5-1),1,EEG.srate*5);                
        
        % Retrive EEG data set
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EEG_dataset_to_merge(end),'study',0); 
        if (~isempty(valid_EMG_onset_detected))
            % Import EMG onset markers 
            EEG = pop_importevent( EEG, 'event',[closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
                '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_EMGonset_markers.txt'],'fields',{'type' 'latency'},'timeunit',1,'align',0);
            EEG = eeg_checkset( EEG );
        else
            %Do nothing
        end

        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_eeg_raw.set'],...
            'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
         % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        EEG_dataset_to_merge(block_index) = CURRENTSET;
    end
        
    if ~isempty(EEG_dataset_to_merge)
        fprintf('All block have been imported. Merging %d blocks...\n', length(EEG_dataset_to_merge));
        if length(EEG_dataset_to_merge) == 1
            % Retrive old data set - No need to merge 
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EEG_dataset_to_merge,'study',0); 
        else
            EEG = pop_mergeset( ALLEEG,EEG_dataset_to_merge, 0);
        end
        EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_eeg_raw'];
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_eeg_raw.set'],'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
    else
        errordlg('Error: EEG blocks could not be merged');
    end
    % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    %    eeglab redraw;

    if ~isempty(EMG_dataset_to_merge)
        fprintf('All block have been imported. Merging %d blocks...\n', length(EMG_dataset_to_merge));
        if length(EMG_dataset_to_merge) == 1
            % Retrive old data set - No need to merge 
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EMG_dataset_to_merge,'study',0); 
        else
            EEG = pop_mergeset( ALLEEG,EMG_dataset_to_merge, 0);
        end
        EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_emg_preprocessed'];
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(total_no_of_trials) '_emg_preprocessed.set'],'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
    else
        errordlg('Error: EMG blocks could not be merged');
    end
        

    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end

%% Preprocessing of raw EEG signals
% % Inputs - SB_raw.set
% % Outputs - SB_preprocessed.set; SB_standardized.set

if process_raw_eeg == 1    
    % ****STEP 0: Load EEGLAB dataset (SB_ses1_cond1_block80_eeg_raw.set)
    % EEGLAB dataset with only EEG channels
    % read in the dataset
    
    EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_eeg_raw.set'], closeloop_folder_path);     
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;
      
    % Define Paramteres
    raw_eeg_Fs = EEG.srate;
    raw_eeg = EEG.data;
    [eeg_nbchns,eeg_pnts] = size(EEG.data);
    
    % ****STEP 1: Plot/Scroll Data in EEGLAB and manually remove noisy
    %         channels/segments of data.

    % Additional STEP: Perform ICA analysis and remove artifacts\
    % ICA commands - icaact(), icaproj(), runica()
    
    % sources = W * channels
    % Unmixing matrix  W = EEG.icaweights*EEG.icasphere
    % W * EEG.icawinv = I, i.e. EEG.icainv is the mixing matrix (Winv)
    % Scalp maps are prepared from column of Winv
    
    % To calculate ICA activations
        %ICAact = icaact(EEG.data,weights*msphere,mean(EEG.data,2));
        
    % Plot component scalp map
%      figure; 
%      topoplot(winv(:,4), EEG.chanlocs,'maplimits', [-3 3],'style','both',...    
%          'electrodes','off','plotrad', 0.5,...
%          'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off',...
%          'conv','off');

        
    % ****STEP 2: High Pass Filter data, 0.1 Hz, 4th order Butterworth
    if use_band_pass == 1
            [b_bpf,a_bpf] = butter(4,([hpfc lpfc]./(raw_eeg_Fs/2)),'bandpass');
            SOS_bpf = tf2sos(b_bpf,a_bpf);
            bpf_df2sos = dfilt.df2sos(SOS_bpf);
            %bpf_df2sos.States = zeros(2,2);
            bpf_df2sos.PersistentMemory = false;
            HPFred_eeg = zeros(size(raw_eeg));

            for i = 1:eeg_nbchns
                if use_noncausal_filter == 1
                    %HPFred_eeg(i,:) = filtfilt(eeg_hpf.sos.sosMatrix,eeg_hpf.sos.ScaleValues,double(raw_eeg(i,:))); % filtering with zero-phase delay
                    HPFred_eeg(i,:) = filtfilt(bpf_df2sos,double(raw_eeg(i,:))); % filtering with zero-phase delay
                else
                    HPFred_eeg(i,:) = filter(bpf_df2sos,double(raw_eeg(i,:)));             % filtering with phase delay 
                end
            end
    else
            if use_fir_filter ==  1
                num_hpf = fir_hp_filter;
                den_hpf = 1;
            else
                [num_hpf,den_hpf] = butter(4,(hpfc/(raw_eeg_Fs/2)),'high');
            end
            
            % DO NOT HIGH-PASS FILTER bcoz amplifier is already filtering - Nikunj 7-11-2017
            HPFred_eeg = double(EEG.data);         
    end
            
    % ****STEP 3: Low Pass Filter data, 1 Hz, 4th order Butterworth
    if use_band_pass == 1
        %LPFred_eeg = SPFred_eeg;
        LPFred_eeg = HPFred_eeg;        % Added 9/4/2015
    else
        if use_fir_filter ==  1
            num_lpf = fir_lp_filter;
            den_lpf = 1;
        else
            [num_lpf,den_lpf] = butter(4,(lpfc/(raw_eeg_Fs/2)));        % IIR Filter
        end
        % num_lpf = fir1(100,(lpfc/(raw_eeg_Fs/2))); den_lpf = 1;       % FIR Filter, Hamming Window, 40th order  
        %eeg_lpf = dfilt.df2(num_lpf,den_lpf);       % Create filter object. Not essential
        %   figure;
        %   freqz(num_lpf,den_lpf,512,raw_eeg_Fs);  % Better to use fvtool
        %   fvtool: http://www.mathworks.com/help/signal/ref/fvtool.html#f7-1176930
        % fvtool(eeg_lpf,'Analysis','freq','Fs',raw_eeg_Fs);  % Essential to visualize frequency response properly.
        LPFred_eeg = zeros(size(HPFred_eeg));       
        for i = 1:eeg_nbchns
            %LPFred_eeg(i,:) = filtfilt(num_lpf,den_lpf,double(EEG.data(i,:)));
            if use_noncausal_filter == 1
                %LPFred_eeg(i,:) = filtfilt(eeg_lpf.sos.sosMatrix,eeg_lpf.sos.ScaleValues,double(SPFred_eeg(i,:))); % filtering with zero-phase delay
                LPFred_eeg(i,:) = filtfilt(num_lpf,den_lpf,double(HPFred_eeg(i,:)));
            else
                %LPFred_eeg(i,:) = filter(num_lpf,den_lpf,double(SPFred_eeg(i,:))); % filtering with phase delay
                LPFred_eeg(i,:) = filter(num_lpf,den_lpf,double(HPFred_eeg(i,:))); % filtering with phase delay
            end
        end
        %      if use_fir_filter == 1
        %         %correct for filter delay
        %         grp_delay = 1000/2; 
        %         LPFred_eeg(:,1:grp_delay) = [];
        %      end
    end        
    
    % Apply ICA here....9/6/2015
    
    % ****STEP 4: Re-Reference data
    SPFred_eeg = LPFred_eeg;
    SPFred_eeg = spatial_filter(SPFred_eeg,spatial_filter_type,[], true);          
    EEG.data = SPFred_eeg;  % Added 9/4/2015    
             
    % **** STEP 5: Resample to 200 Hz 
    %Preproc_eeg = decimate(LPFred_eeg,5);    
    EEG = pop_resample(EEG, 200);
    EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_preprocessed'];
    EEG = eeg_checkset( EEG );
    Fs_eeg = EEG.srate;           % Sampling rate after downsampling
    
    % **** STEP 5A: Remove extra stimulus triggers 'S  2'
    % Deletes 'S  2' trigger for trials which were aborted
    deleted_rest_trig_latency = [];
    for j=1:length(EEG.event)-1
        if (strcmp(EEG.event(j).type,rest_trig_label))
            %if(strcmp(EEG.event(j+1).type,rest_trig_label))
            if(strcmp(EEG.event(j+1).type,'S 32'))      % Added 9-2-2015
                EEG.event(j).type = 'DEL';
                deleted_rest_trig_latency = [deleted_rest_trig_latency; EEG.event(j).latency/Fs_eeg];                               
            end
        end
    end     
    EEG = eeg_checkset( EEG );
    
%     % **** STEP 6: Import EMG onset markers 
%     EEG = pop_importevent( EEG, 'event',[closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_EMGonset_markers.txt'],'fields',{'type' 'latency'},'timeunit',1,'align',0);
%     EEG = eeg_checkset( EEG );

    % Save dataset    
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_eeg_preprocessed.set'],'filepath',closeloop_folder_path);    
 
    %Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    % **** STEP -: Compute Power Spectral Density of EEG Channels

%     NFFT = 512; % Freq_Res = Fs/NFFT - This is true !!
%     PSD_eeg = zeros(length(Channels_nos),((NFFT/2)+1));
%     PSD_f = zeros(1,((NFFT/2)+1));      % range = (-fs/2:0:fs/2). But PSD is real and symmetric. Hence only +ve frequencies plotted
%     figure; hold on; grid on;
%     for psd_len = 1:length(Channels_nos)
%         [PSD_eeg(psd_len,:),PSD_f] = pwelch(detrend(double(EEG.data(Channels_nos(psd_len),:))),[],[],NFFT,Fs_eeg);   
%         plot(PSD_f(:),10*log10(PSD_eeg(psd_len,:)),myColors(psd_len),'LineWidth',1.5);
%         hold on;
%     end
%     xlabel('Frequency Hz');
%     ylabel('PSD dB');
%     title('Preprocessed EEG Channels PSD');
%     hold off
    
    % **** STEP 7: Standardize EEG data
    % Standardization/Normalization of Signals; 
    % http://en.wikipedia.org/wiki/Standard_score
    % Standard score or Z-score is the number of standard deviations an
    % obervation is above/below the mean. 
    % z-score = (X - mu)/sigma;
% %     if standardize_data_flag == 1
% %         [standardized_eeg,mu_eeg,sigma_eeg] = zscore(EEG.data');
% %         EEG.data = standardized_eeg';
% %         EEG.setname=[Subject_name '_standardized'];
% %         EEG = eeg_checkset( EEG );
% %         % Save dataset
% %         EEG = pop_saveset( EEG, 'filename',[Subject_name '_standardized.set'],'filepath',folder_path);
% %         EEG = eeg_checkset( EEG );
% %         
% %         % Update EEGLAB window
% %         [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
% %         eeglab redraw;
% %     end

end

%% Extract move epochs and rest epochs
if extract_epochs == 1
    
    %% Extract move epochs
    EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_eeg_preprocessed.set'], closeloop_folder_path);    
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    %eeglab redraw;        
       
    EEG = pop_epoch( EEG, {  move_trig_label  }, move_epoch_dur, 'newname', 'EMGmove_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    
    if ~isempty(remove_corrupted_epochs)
        reject_epochs = zeros(1,size(EEG.epoch,2));
        reject_epochs(remove_corrupted_epochs) = 1;
        EEG = pop_rejepoch(EEG,reject_epochs,1);
    else
%         pop_eegplot(pop_select(EEG,'channel',Channels_nos(6:end)),1,1,0)
    end

    EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_move_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_move_epochs.set'],'filepath',closeloop_folder_path);    
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end

%% Manipulate & Calculate ERPs for Move Epochs
if manipulate_epochs == 1            
    EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_move_epochs.set'], closeloop_folder_path);
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

            %1. Subtract mean value of rest from rest trials
            if apply_baseline_correction == 1 
                move_epochs(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt) - move_mean_baseline(channel_cnt,epoch_cnt);
            end
            move_epochs_with_base_correct(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt) - move_mean_baseline(channel_cnt,epoch_cnt);
            %move_epochs(epoch_cnt,:,channel_cnt) = log(abs(move_epochs(epoch_cnt,:,channel_cnt) - move_mean_baseline(channel_cnt,epoch_cnt)));           
        end
    end

    % If after baseline correction the range of the signal is too high then
    % delete that epoch    
    epochs_with_artefacts = find(range(move_epochs_with_base_correct(:,:,14),2) >= 50);
    move_epochs_with_base_correct(epochs_with_artefacts, :, :) = [];

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

%% Plot ERPs
if plot_ERPs == 1
    paper_font_size = 10;
%     figure('Position',[100 100 3.5*116 2.5*116]); 
    figure;
    %figure('units','normalized','outerposition',[0 0 1 1])
    T_plot = tight_subplot(numel(Channels_nos)/7,7,[0.01 0.01],[0.15 0.01],[0.1 0.1]);
    hold on;
    plot_ind4 = 1; % Index of plot where axis should appear
        
    for ind4 = 1:length(Channels_nos)
        axes(T_plot(ind4));
%         if plot_ind4 == 5                 % Commented by Nikunj on Oct 31,2013
%             plot_ind4 = plot_ind4 + 1;
%         end
        %subplot(5,5,plot_ind4);
        hold on;
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:),'k','LineWidth',1.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:) - (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
        
        % Added 8/21/2015        
        text(-2,-4,[EEG.chanlocs(Channels_nos(ind4)).labels ',' int2str(Channels_nos(ind4))],'Color','k','FontWeight','normal','FontSize',paper_font_size-1); % ', ' num2str(Channels_nos(ind4))
        set(gca,'YDir','reverse');
        %if max(abs(move_avg_channels(Channels_nos(RPind),:))) <= 6
%          if max((move_avg_channels(Channels_nos(ind4),:)+(move_SE_channels(Channels_nos(ind4),:)))) >= 6 || ...
%                  min((move_avg_channels(Channels_nos(ind4),:)-(move_SE_channels(Channels_nos(ind4),:)))) <= -6
%             axis([move_erp_time(1) move_erp_time(end) -15 15]);            
%             set(gca,'FontWeight','bold');  
%             set(gca,'YTick',[-10 0 10]);
%             set(gca,'YTickLabel',{'-10'; '0'; '10'});
%         else
            axis([move_epoch_dur(1) move_epoch_dur(2) -10 10]);
            %axis([move_erp_time(1) 1 -15 15]);
            % set(gca,'YTick',[-5 0 2]);                                                                                                                                            % uncomment for publication figure
            % set(gca,'YTickLabel',{'-5'; '0'; '+2'},'FontWeight','normal','FontSize',paper_font_size-1);                        % uncomment for publication figure
%        end
        line([0 0],[-30 20],'Color','k','LineWidth',0.5,'LineStyle','--');  
        line([-2.5 4],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
        plot_ind4 = plot_ind4 + 1;
        grid on;
        
        if ind4 == 29
            % set(gca,'XColor',[1 1 1],'YColor',[1 1 1])          % uncomment for publication figure
            % set(gca,'YtickLabel',' ');
             set(gca,'YTick',[-10 -5 0 5]);                                                                                                                                            % comment for publication figure
            set(gca,'YTickLabel',{'-10';'-5'; '0'; '+5'},'FontWeight','normal','FontSize',paper_font_size-1);                        % comment for publication figure
             hylab = ylabel('MRCP Grand Average','FontSize',paper_font_size-1,'Color',[0 0 0]);
             pos_hylab = get(hylab,'Position');
             set(hylab,'Position',[pos_hylab(1) pos_hylab(2) pos_hylab(3)]);
        else
            set(gca,'Visible','on');        % change to 'off' for publication figure
        end
        
    %    grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
     %   set(gca,'XTick',[-2 -1 0 1]);
     %   set(gca,'XTickLabel',{'-2';'-1'; '0';'1'});  
        
          
    end
       
    axes(T_plot(29));
    set(gca,'Visible','on');
    bgcolor = get(gcf,'Color');
    set(gca,'YColor',[0 0 0]);
    set(gca,'XTick',[-2 -1 0 1]);
    set(gca,'XTickLabel',{'-2'; '-1'; 'MO';'1'},'FontSize',paper_font_size-1);  
    set(gca,'TickLength',[0.03 0.025])
    hold on;
    xlabel('Time (s)', 'FontSize',paper_font_size-1);
    
    % Annotate line
    axes(T_plot(35));
    axes_pos = get(gca,'Position'); %[lower bottom width height]
    axes_ylim = get(gca,'Ylim');
    annotate_length = (5*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
    annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.025) (axes_pos(1)+axes_pos(3)+0.025)],...
        [(axes_pos(2)+axes_pos(4) - annotate_length/5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/5)],'LineWidth',2);
    
    %hylabel = ylabel('EEG (\muV)','FontSize', 10, 'rotation',90);
    %pos = get(hylabel,'Position');
    
    %mtit(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'fontsize',14,'color',[0 0 0],'xoff',-.02,'yoff',.025);
    %mtit('LSGR, Left hand, Triggered Mode, Day 1','fontsize',14,'yoff',0.025);
    %title(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)));
    %legend('Average RP','Movement Onset','Orientation','Horizontal');
    %export_fig MS_ses1_cond1_block80_Average '-png' '-transparent';
    
    % Expand axes to fill figure
%     fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
%     hgexport(fig,'-clipboard',style,'applystyle', true);
%     drawnow;
    
%     response = input('Save figure to folder [y/n]: ','s');
%     if strcmp(response,'y')
%          %tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.tif'];
%          %fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.fig'];
%         %print('-dtiff', '-r300', tiff_filename); 
%         %saveas(gcf,fig_filename);

%     else
%         disp('Save figure aborted');
%     end
    
    %% ERP image plots
%     mt1 = 101;  
%     peak_times = move_erp_time(mt1 + min_avg(:,2));
%     [sorted_peak_times,sorting_order] = sort(peak_times,'descend');
%     modified_EEG_data = EEG.data(:,:,sorting_order);
%     EEG.data = modified_EEG_data;
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
%     eeglab redraw;
    
    %Channels_nos = [9 32 10 44 48 14 49 15 19 53 20 54];
%     ra1 = [-15 15];
%     figure;
%     T_plot = tight_subplot(numel(Channels_nos)/7,7,0.05,[0.1 0.1],[0.1 0.1]);
%     for ind4 = 1:length(Channels_nos)
%         axes(T_plot(ind4));
%         if ind4 == 35
%             pop_erpimage(EEG,1, Channels_nos(ind4),[[]],'',5,1,{},[],'','noxlabel','cbar','on','caxis',[ra1(1) ra1(2)],'cbar_title','\muV'); %EEG.chanlocs(Channels_sel(ch)).labels
%             
%         else
%             pop_erpimage(EEG,1, Channels_nos(ind4),[[]],'',5,1,{},[],'','noxlabel','cbar','off','caxis',[ra1(1) ra1(2)],'cbar_title','\muV'); %EEG.chanlocs(Channels_sel(ch)).labels
%         end
%     end
%     set(gca,'XTick',[-2 -1 0 1 2 3]);
%     set(gca,'XTickLabel',{'-2'; '-1'; '0';'1';'2';'3'});  
    %export_fig MS_ses1_cond1_block80_ERP '-png' '-transparent';
    %print -dtiff -r450 LSGR_cond3_erp.tif
    %mtit([Subject_name ', Backdrive Mode, Day 1 & 2'],'fontsize',14,'yoff',0.025);
    
    %% Comparing plots
% %     % mychanns = [9;10;48;14;49];
% %     mychanns = Channels_nos(1:9);
% %     figure;
% %     hold on;
% %     plot_ind5 = 1;
% %     for RPind = 1:length(mychanns)
% % %         if plot_ind5 == 2
% % %            plot_ind5 = plot_ind5 + 1;
% % %         end
% %         subplot(3,3,plot_ind5);hold on;
% %         %plot(self_ini_time,self_ini(mychanns(plot_cnt),:),'r',robo_ini_time,robo_ini(mychanns(plot_cnt),:),'b',robo_ini_time1,robo_ini1(mychanns(plot_cnt),:),'g','LineWidth',2);
% %         %plot(self_ini_time,self_ini(mychanns(plot_cnt),:),'r',robo_ini_time,robo_ini(mychanns(plot_cnt),:),'b','LineWidth',2);
% %         %plot(self_ini_time,self_ini(mychanns(plot_cnt),:),'r','LineWidth',2);
% %         plot(move_erp_time,move_avg_channels(mychanns(RPind),:),'b','LineWidth',2);
% %         jbfill(move_erp_time,move_avg_channels(mychanns(RPind),:)+move_std_channels(mychanns(RPind),:),...
% %             move_avg_channels(mychanns(RPind),:)-move_std_channels(mychanns(RPind),:),'b','b',0,0.3);
% %         plot(rest_erp_time,rest_avg_channels(mychanns(RPind),:),'r','LineWidth',2);
% %         jbfill(rest_erp_time,rest_avg_channels(mychanns(RPind),:)+rest_std_channels(mychanns(RPind),:),...
% %             rest_avg_channels(mychanns(RPind),:)-rest_std_channels(mychanns(RPind),:),'r','k',0,0.15);
% % 
% %         text(-1.5,5,['Channel ' EEG.chanlocs(mychanns(RPind)).labels],'Color','k');
% %         set(gca,'YDir','reverse');
% %         xlabel('Time (sec)');
% %         ylabel('\mu Volts');
% %         axis([move_erp_time(1) move_erp_time(end) -15 15]);
% %         line([0 0],[-10 10],'Color','k','LineWidth',2);
% %         %text(0.2,8,'Movement Onset','Rotation',90,'FontSize',8);
% %     %     if plot_cnt == 2
% %     %         %legend('Self Init, ERP 0.1 - 1Hz','Robot Init, ERP 0.1 - 1Hz','Robot Init, ERP 0.1 - 5Hz');
% % 
% %     %     end
% %         plot_ind5 = plot_ind5 + 1;
% %     end
% %     %legend('Avg. RP during Movement','Variations in RP during Movement',...
% %     %          'Avg. RP during Rest','Variations in RP during Rest','Onset of Movement');
end

%% Calculate features based on RPs
if Performance.use_smart_features == 1
    
    Fs_eeg_ds = 20; % Desired sampling frequency
    downsamp_factor = Fs_eeg/Fs_eeg_ds;    
    % Downsample the epochs and epoch_times
    for k = 1:no_channels
        move_epochs_s(:,:,k) = downsample(move_epochs(:,:,k)',downsamp_factor)';        
    end
    [no_epochs,no_datapts,no_channels] = size(move_epochs_s);
    move_erp_time_ds = downsample(move_erp_time(:),downsamp_factor);
   
%     smart_move_ch_avg = [];
%     move_ch_avg_time = [];
%     smart_rest_ch_avg = [];
%     rest_ch_avg_time = [];
    bad_move_trials = [];
    good_move_trials = [];

    mt1 = find(move_erp_time_ds == Performance.find_peak_interval(1));
    mt2 = find(move_erp_time_ds == Performance.find_peak_interval(2));
    
    move_ch_avg_ini = mean(move_epochs_s(:,:,Performance.optimized_channels),3);                
    [min_avg(:,1),min_avg(:,2)] = min(move_ch_avg_ini(:,mt1:mt2),[],2); % value, indices
    
    for nt = 1:size(move_ch_avg_ini,1)
        if (move_erp_time_ds(move_ch_avg_ini(nt,:) == min_avg(nt,1)) <= Performance.reject_trial_before) %|| (min_avg(nt,1) > -3)        
            %plot(move_erp_time_ds(1:26),move_ch_avg_ini(nt,1:26),'r'); hold on;
            bad_move_trials = [bad_move_trials; nt];
        else
            %plot(move_erp_time_ds(1:26),move_ch_avg_ini(nt,1:26)); hold on;
            good_move_trials = [good_move_trials; nt];
        end
    end
        
%     figure; hold on;
%     plot(move_erp_time_ds,move_ch_avg_ini(good_move_trials,:),'b');
%     title([Subject_name ', # Good trials = ' num2str(length(good_move_trials)) ' (' num2str(size(move_ch_avg_ini,1)) ')'],'FontSize',12);
% %     plot(move_erp_time_ds,move_ch_avg_ini(bad_move_trials,:),'r');
%     set(gca,'YDir','reverse');
%     grid on;
% %     axis([-3.5 1 -15 10]);
                 

    if Performance.keep_bad_trials == 1
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
        move_window_start = move_window_end - Performance.smart_window_length*Fs_eeg_ds; 
%         rest_window_start = find(rest_erp_time == rest_window(1));
%         rest_window_end = rest_window_start + window_length;

        smart_move_ch_avg(i,:) = move_ch_avg_ini(good_move_trials(i),move_window_start:move_window_end);
        move_ch_avg_time(i,:) = move_erp_time_ds(move_window_start:move_window_end);
%         smart_rest_ch_avg(i,:) = rest_ch_avg_ini(good_move_trials(i),rest_window_start:rest_window_end); 
%         rest_ch_avg_time(i,:) = rest_erp_time(rest_window_start:rest_window_end);


%        subplot(2,1,1); hold on;
       plot(move_ch_avg_time(i,:),smart_move_ch_avg(i,:),'k','LineWidth',2); hold on;   
%        subplot(2,1,2); hold on;
%        plot(rest_ch_avg_time(i,:),smart_rest_ch_avg(i,:),'r','LineWidth',2); hold on;
        
    end

    % Reinitialize
    no_epochs = size(smart_move_ch_avg,1);        
%         data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

    %1. Slope
    Smart_Features = [(smart_move_ch_avg(:,end) - smart_move_ch_avg(:,1))./(move_ch_avg_time(:,end) - move_ch_avg_time(:,1))];
%                           (smart_rest_ch_avg(:,end) - smart_rest_ch_avg(:,1))./(rest_ch_avg_time(:,end) - rest_ch_avg_time(:,1))];

    %2. Negative Peak 
    Smart_Features = [Smart_Features [min(smart_move_ch_avg,[],2)]]; %min(smart_rest_ch_avg,[],2)]];

    %3. Area under curve
    for ind1 = 1:size(smart_move_ch_avg,1)
        AUC_move(ind1) = trapz(move_ch_avg_time(ind1,:),smart_move_ch_avg(ind1,:));
%             AUC_rest(ind1) = trapz(rest_ch_avg_time(ind1,:),smart_rest_ch_avg(ind1,:));
    end
%         Smart_Features = [Smart_Features [AUC_move';AUC_rest']];
    Smart_Features = [Smart_Features [AUC_move']];

    %6. Mahalanobis distance of each trial from average over trials           
    % Direct computation of Mahalanobis distance
%         smart_mahal_dist = zeros(2*no_epochs,1);
    smart_mahal_dist = zeros(no_epochs,1);
%         smart_Cov_Mat = cov(smart_move_ch_avg);
%         smart_Mu_move = mean(smart_move_ch_avg,1);
    for d = 1:no_epochs
        x = smart_move_ch_avg(d,:);
        smart_mahal_dist(d) = sqrt((x-Performance.smart_Mu_move)/(Performance.smart_Cov_Mat)*(x-Performance.smart_Mu_move)');
%             y = smart_rest_ch_avg(d,:);
%             smart_mahal_dist(d + no_epochs) = sqrt((y-smart_Mu_move)/(smart_Cov_Mat)*(y-smart_Mu_move)');
    end
    Smart_Features = [Smart_Features smart_mahal_dist];                       
end    

%% Save data for training classifier
if manipulate_epochs == 1
   Posthoc_Average.move_epochs = move_epochs;
%    Posthoc_Average.rest_epochs = rest_epochs;
   Posthoc_Average.Fs_eeg = Fs_eeg;
   Posthoc_Average.apply_baseline_correction = apply_baseline_correction;

   Posthoc_Average.move_avg_channels = move_avg_channels;
   Posthoc_Average.move_std_channels = move_std_channels;
   Posthoc_Average.move_SE_channels = move_SE_channels;
   Posthoc_Average.move_mean_baseline = move_mean_baseline;   
   Posthoc_Average.move_erp_time = move_erp_time;
   Posthoc_Average.epochs_with_artefacts = epochs_with_artefacts;

%    Average.rest_avg_channels = rest_avg_channels;
%    Average.rest_std_channels = rest_std_channels;
%    Average.rest_SE_channels = rest_SE_channels;
%    Average.rest_mean_baseline = rest_mean_baseline;   
%    Average.rest_erp_time = rest_erp_time;
   Posthoc_Average.remove_corrupted_epochs = remove_corrupted_epochs;
   
%    disp('Suggested RP channels = '); disp(Channels_criteria1_and_2);
%    Average.RP_chans = input('Enter channel numbers where RP is seen. Example - [1 2 3 4]: ');
   Posthoc_Average.RP_chans = Performance.classchannels;
   Posthoc_Average.optimized_channels = Performance.optimized_channels;   
   Posthoc_Average.bad_move_trials = bad_move_trials;
   Posthoc_Average.good_move_trials = good_move_trials;
   Posthoc_Average.smart_window_length = Performance.smart_window_length;   
   Posthoc_Average.Smart_Features = Smart_Features;       % New Features
   
   filename1 = [closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(Block_num) '_posthoc_average.mat'];
   save(filename1,'Posthoc_Average');   
end
end