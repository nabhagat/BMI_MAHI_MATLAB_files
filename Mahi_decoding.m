% Program for processing EEG, EMG and Kinematics data collected for the
% BMI-Mahi Project

% Created By:  Nikunj Bhagat, Graduate Student, University of Houston
% Contact : nbhagat08[at]gmail.com
% Date Created - 1/23/2014

%% ***********Revisions 
% 2/08/14 - Replacing EEGLAB functions.
%         - No standardization, extract targets option. 
% 5/19/14 - Using filter() instead of filtfilt() for phase-delayed low-pass and
%           high-pass filtering. Required for trained classifier to work properly
%           during realtime. 
% 5/22/14 - Adding Subversioning using TortoiseSVN. The repository is
%           called SVN_Repository_Nikunj_Data
%         - Removing code added for phase randomization
% 5/31/14 - Raster plot after completion of closed-loop trial
% 6/03/14 - Saving Position, Velocity & Torque epochs for each block. Epch
%           duration [-2.5s 6s]. Also removing epochs i.e. trials with 
%           corrupted EEG signals. No baseline correction for plots.
% 6/07/14 - Adding option to choose between causal(filter) and noncausal(filtfilt)
%           filters. Noncausal filters required for conventional filters
%         - Removed option use_separate_move_rest_epochs. Always save
%           separate move & rest epcohs.
% 6/10/14 - Changed baseline interval to [-2.5 -2.25]s - Later baseline
%           correction was completely removed 
% 6/11/14 - Replace extra 'S  2' Trigger labels (Aborted Trials) with
%           'DEL' label
% 7/09/14 - Added function to call GUI and pass user information and
%           serial_obj to GUI.
% 7/14/14 - Added if/ese to switch between calling GUI or script for
%           closeloop testing.
% 10/10/14 - Save remove_corrupted_epochs in Average cell structure
% ?       - Need to add SVM training into this file, train_classifier
% 11/4/14 - Added instructions to save emg_rest_epochs in the emg_epochs
%                    variable
% 1/2/15 - Changed baseline interval from [-3.5 -3.25] to [-2.5 -2.25].
%                  Doesnot affect analysis because baseline correction is
%                  only used for visualizing grand-average MRCP
% 1/27/15 - Calculate t-value for 95% confidence interval using the number
%                    of trials 
%--------------------------------------------------------------------------------------------------
clear;
%close all;
%% Global variables 
myColors = ['r','b','k','m','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

% EEG Channels used for identifying MRCP
Channels_nos = [43, 9, 32, 10, 44, 13, 48, 14, 49, 15, 52, 19, ... 
                53, 20, 54]; % removed P-channels = [24, 57, 25, 58, 26]; removed F-channels = [4, 38, 5, 39, 6];    % 32 or 65 for FCz

% Subject Details 
Subject_name = 'BNBO'; % change1
Sess_num = '2';               
closeloop_Sess_num = '6';     
Cond_num = 3;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;

folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
closeloop_folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3
%folder_path = ['F:\Nikunj_Data\InMotion_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
           
% Flags to control the processing 
train_classifier = 1;   %change4
test_classifier = 0;
use_GUI_for_testing = 1;    %change5

if train_classifier == 1
    disp('******************** Training Model **********************************');
    process_raw_eeg = 0;         % Also remove extra 'S  2' triggers
    process_raw_emg = 0; extract_emg_epochs = 0;
    extract_epochs = 0;     % extract move and rest epochs
  
    % Used during extracting epochs for removing corrupted epochs. The numbers of corrupted epochs 
    % must be known in advance. Otherwise declare remove_corrupted_epochs = [];
    
    remove_corrupted_epochs = [];
    %remove_corrupted_epochs = [ remove_corrupted_epochs 41 125 153]; %ERWS_ses2_cond3_block160

    %remove_corrupted_epochs = [ remove_corrupted_epochs 21 42 111 112 140 147]; %PLSH_ses2_cond1_block160

    %remove_corrupted_epochs = [ remove_corrupted_epochs 41 43 52 54 59]; %LSGR_ses1_cond3
    %remove_corrupted_epochs = [remove_corrupted_epochs 83];  % LSGR_ses2b_cond3
   
   %remove_corrupted_epochs = [41];  % JF_ses1_cond1
    
   %remove_corrupted_epochs = [61 74];
    %remove_corrupted_epochs = [21 25 28 29 30 31 36 37 39 40 61]; % MS_ses1_cond1
    %remove_corrupted_epochs = [80]; % MS_ses2_cond2
    %remove_corrupted_epochs = [7 8 10 11 12 15 17 18 19 20]; % MS_ses1_cond3
    %remove_corrupted_epochs = [40 76 79 80]; % MS_ses2_cond4
       
    %remove_corrupted_epochs = [1 15];   % TA_ses1_cond1
    %remove_corrupted_epochs = [13 16 29 46 65];   % TA_ses1_cond2
    %remove_corrupted_epochs = [12 14 29 40 48 49 72 74];   % TA_ses1_cond3
    %remove_corrupted_epochs = [2 6 7 9 15 16 39 42 70]; % TA_ses1_cond4
    
    %remove_corrupted_epochs = [20 60]; %CR_ses1_cond1
    %remove_corrupted_epochs = [4 16 21 22 26 31 32 36 40 51 58 59 65 66 80]; %CR_ses1_cond2
    %remove_corrupted_epochs = []; %CR_ses1_cond3
    %remove_corrupted_epochs = [33 59 61 66 75 80]; %CR_ses1_cond4
    
    %remove_corrupted_epochs = [20 57 58 4]; %MR_ses2_cond1
    %remove_corrupted_epochs = [20]; %MR_ses2_cond2
    %remove_corrupted_epochs = [21 22 50 60]; %MR_ses1_cond3
    %remove_corrupted_epochs = [49 50 61]; %MR_ses1_cond4
    
    manipulate_epochs = 1;
    plot_ERPs = 1;
    label_events = 0;       % process kinematics and label the events/triggers
    use_kinematics_old_code = 0;    % Use old code for InMotion
    kin_blocks = [1 2 3 4;                % Session 1
                             1 2 3 4];              % Session 2
     % kin_blocks must have 4 columns. If a block is absent, replace it with 0. But don't replace zero for first and last block.   
    
    use_phase_rand = 0;     % For empirically estimating chance levels. Currently not used
   %standardize_data_flag = 0;     % Not used
    include_target_events = 0;      % Not used
       
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 
    
elseif test_classifier == 1
    disp('******************** Close-loop Model testing *************************');
    process_raw_eeg = 0;
    process_raw_emg = 0;
    extract_epochs = 0;     % extract move and rest epochs
    manipulate_epochs = 0;
    plot_ERPs = 0;
    label_events = 0;       % process kinematics and label the events/triggers
    extract_emg_epochs = 0;
    use_phase_rand = 0;
   %standardize_data_flag = 0;
    include_target_events = 0;      
end
% Pre-processing Flags
%1. Spatial Filter Type
    large_laplacian_ref  = 1;
    common_avg_ref = 0; car_chnns_eliminate = [1 2 41 46]; % car_chnns_eliminate = [];  
    small_laplacian_ref = 0;
    weighted_avg_ref = 0;
    bipolar_ref    = 0;
    
    spatial_filter_type = 'none';
    if common_avg_ref ==  1
       spatial_filter_type = 'CAR';
    end
    if small_laplacian_ref ==  1
       spatial_filter_type = 'SLAP';
    end
    if large_laplacian_ref ==  1
       spatial_filter_type = 'LLAP';
    end
    if weighted_avg_ref ==  1
       spatial_filter_type = 'WAVG';
    end
    
%2. Filter cutoff frequency  
hpfc = 0.1;     % HPF Cutoff frequency = 0.1 Hz    
lpfc = 1;      % LPF Cutoff frequency = 1 Hz
use_noncausal_filter = 0; % 1 - yes, use zero-phase filtfilt(); 0 - No, use filter()            %change6
use_fir_filter = 0; % change7

%3. Extracting epochs
move_trig_label = 'S 16';  % 'S 32'; %'S  8'; %'100';
rest_trig_label = 'S  2';  % 'S  2'; %'200';
target_reached_trig_label = 'S  8';
move_epoch_dur = [-7 1.6]; % [-4 12.1];
rest_epoch_dur = [-3.5 1.6];

%4. Working with Epochs
baseline_int = [-2.5 -2.25];   
apply_baseline_correction = 0;  % Always 0
apply_epoch_standardization = 0;

%5. Which Classifier to use?
train_LDA_classifier = 0;           % Not used
train_SVM_classifier = 1;

%6. Segment data for predicting kinematics
segment_data_for_decoding_kinematics = 0;
%% Preprocessing of raw EEG signals
% % Inputs - SB_raw.set
% % Outputs - SB_preprocessed.set; SB_standardized.set

if process_raw_eeg == 1    
    % ****STEP 0: Load EEGLAB dataset (SB_ses1_cond1_block80_eeg_raw.set)
    % EEGLAB dataset with only EEG channels
    % read in the dataset
    if ~isempty(Cond_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_raw.set'], folder_path); 
    elseif ~isempty(Block_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_raw.set'], folder_path); 
    else
        EEG = pop_loadset( [Subject_name '_raw.set'], folder_path); 
    end
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
    if use_fir_filter ==  1
        num_hpf = fir_hp_filter;
        den_hpf = 1;
    else
        [num_hpf,den_hpf] = butter(4,(hpfc/(raw_eeg_Fs/2)),'high');
    end
    %eeg_hpf = dfilt.df2(num_hpf,den_hpf);       % Create filter object. Not essential  
    %   figure;
    %   freqz(num_hpf,den_hpf,512,raw_eeg_Fs); % Better to use fvtool
    %   fvtool(eeg_hpf,'Analysis','freq','Fs',raw_eeg_Fs);  % Essential to visualize frequency response properly.
    HPFred_eeg = zeros(size(raw_eeg));
    for i = 1:eeg_nbchns
        %EEG.data(i,:) = detrend(EEG.data(i,:)); %Commented on 12-9-13
        %HPFred_eeg(i,:) = filtfilt(num_hpf,den_hpf,double(EEG.data(i,:)));
        
        if use_noncausal_filter == 1
            %HPFred_eeg(i,:) = filtfilt(eeg_hpf.sos.sosMatrix,eeg_hpf.sos.ScaleValues,double(raw_eeg(i,:))); % filtering with zero-phase delay
            HPFred_eeg(i,:) = filtfilt(num_hpf,den_hpf,double(raw_eeg(i,:))); % filtering with zero-phase delay
        else
            HPFred_eeg(i,:) = filter(num_hpf,den_hpf,double(raw_eeg(i,:)));             % filtering with phase delay 
        end
    end
    %EEG.data = HPFred_eeg;

%     if use_fir_filter == 1
%         %correct for filter delay
%         grp_delay = 10000/2; 
%         HPFred_eeg(:,1:grp_delay) = [];
%     end
    
    % ****STEP 3: Re-Reference data
    SPFred_eeg = HPFred_eeg;
    
    if common_avg_ref == 1
%         %EEG = pop_reref( EEG, [],'exclude',[1 2 17]);       % Exclude HEOG, VEOG channels
%         EEG = pop_reref( EEG, []);
%         % EEG.setname='NB_CAR';
%         EEG = eeg_checkset( EEG );
        
%         % Create Common Avg Ref matrix of size 64x64
%         total_chnns = eeg_nbchns;
%         num_car_channels = total_chnns - length(car_chnns_eliminate);          % Number of channels to use for CAR
%         M_CAR =  (1/num_car_channels)*(diag((num_car_channels-1)*ones(total_chnns,1)) - (ones(total_chnns,total_chnns)-diag(ones(total_chnns,1))));
%         
%         if ~isempty(car_chnns_eliminate)
%             for elim = 1:length(car_chnns_eliminate)
%                 M_CAR(:,car_chnns_eliminate(elim)) = 0;
%             end
%         end
%         
%         SPFred_eeg = M_CAR*(SPFred_eeg);

    SPFred_eeg = spatial_filter(SPFred_eeg,'CAR',car_chnns_eliminate);
    end 
    if large_laplacian_ref == 1
%         EEG = exp_eval(flt_laplace(EEG,4));
%         EEG.ref = 'laplacian';
%         EEG.setname = 'laplacian-ref';
%         EEG = eeg_checkset( EEG );      
    SPFred_eeg = spatial_filter(SPFred_eeg,'LLAP',[]);          
    end 
    if small_laplacian_ref == 1
    SPFred_eeg = spatial_filter(SPFred_eeg,'SLAP',[]);    
    end
    if weighted_avg_ref == 1
    SPFred_eeg = spatial_filter(SPFred_eeg,'WAVG',[]);
    end
        
    % ****STEP 4: Low Pass Filter data, 1 Hz, 4th order Butterworth
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
    LPFred_eeg = zeros(size(SPFred_eeg));
    for i = 1:eeg_nbchns
        %LPFred_eeg(i,:) = filtfilt(num_lpf,den_lpf,double(EEG.data(i,:)));
        if use_noncausal_filter == 1
            %LPFred_eeg(i,:) = filtfilt(eeg_lpf.sos.sosMatrix,eeg_lpf.sos.ScaleValues,double(SPFred_eeg(i,:))); % filtering with zero-phase delay
            LPFred_eeg(i,:) = filtfilt(num_lpf,den_lpf,double(SPFred_eeg(i,:)));
        else
            LPFred_eeg(i,:) = filter(num_lpf,den_lpf,double(SPFred_eeg(i,:))); % filtering with phase delay
        end
    end
%      if use_fir_filter == 1
%         %correct for filter delay
%         grp_delay = 1000/2; 
%         LPFred_eeg(:,1:grp_delay) = [];
%      end
    
    EEG.data = LPFred_eeg;   
    
     
             
    % **** STEP 5: Resample to 100 Hz 
    %Preproc_eeg = decimate(LPFred_eeg,5);    
    EEG = pop_resample( EEG, 200);
    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_preprocessed'];
    EEG = eeg_checkset( EEG );
    Fs_eeg = EEG.srate;           % Sampling rate after downsampling
    
    % **** STEP 5A: Remove extra stimulus triggers 'S  2'
    % Deletes 'S  2' trigger for trials which were aborted
    deleted_rest_trig_latency = [];
    for j=1:length(EEG.event)-1
        if (strcmp(EEG.event(j).type,rest_trig_label))
            if(strcmp(EEG.event(j+1).type,rest_trig_label))
                EEG.event(j).type = 'DEL';
                deleted_rest_trig_latency = [deleted_rest_trig_latency; EEG.event(j).latency/Fs_eeg];                               
            end
        end
    end     
    EEG = eeg_checkset( EEG );
    % Save dataset
    if ~isempty(Cond_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'],'filepath',folder_path);    
    elseif ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_preprocessed.set'],'filepath',folder_path);
    end
    
    %Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    % **** STEP 6: Compute Power Spectral Density of EEG Channels

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
%% Preprocess EMG signals
if process_raw_emg == 1
    % EEGLAB dataset with only EMG channels (SB_ses1_cond1_block80_emg_raw.set)
    if ~isempty(Cond_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_raw.set'], folder_path); 
    else
        error('No EMG file available. Check!!');
    end 
    %Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    myColors = ['r','b','k','m','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];

    emg_Fs = EEG.srate;
    raw_emg = EEG.data;
    raw_emg_t = 0:1/emg_Fs: (length(raw_emg) - 1)/emg_Fs;
    
    emg_for_psd = raw_emg;
    %emg_for_psd = Diff_emg(1,1:1e5);
%     %Check PSD of raw emg signals
    NFFT = 512; % Freq_Res = Fs/NFFT
    PSD_emg = zeros(size(emg_for_psd,1),((NFFT/2)+1));
    PSD_f = zeros(1,((NFFT/2)+1));
    figure; 
    hold on; grid on;
    for psd_len = 1:size(emg_for_psd,1)
        [PSD_emg(psd_len,:),PSD_f] = pwelch(detrend(emg_for_psd(psd_len,:)),[],[],NFFT,emg_Fs);   
        plot(PSD_f(:),10*log10(PSD_emg(psd_len,:)),myColors(psd_len));
        hold on;
    end
    xlabel('Frequency Hz');
    ylabel('PSD dB');
    title('Raw EMG Channels PSD');
    hold off
    
    %figure; pmtm(raw_emg(1,10000:18000),4,512,1000);
    
    % ****STEP 1: Apply Notch filter at 5.859 Hz
    
    
    % ****STEP 2: Bandpass Filter EMG signal, 20 - 45 Hz, 4th order Butterworth
    BPFred_emg = [];
    Diff_emg = [];
    EMG_rms = [];
    
    emgbpfc = [30 200];     % Cutoff frequency = 15 - 48 Hz
    [num_bpf,den_bpf] = butter(4,[emgbpfc(1)/(emg_Fs/2) emgbpfc(2)/(emg_Fs/2)]);
    %emg_bpf = dfilt.df2(num_bpf,den_bpf);       % Create filter object. Not essential  
    % % figure;
    % % freqz(num_bpf,den_bpf,512,emg_Fs); % Better to use fvtool
    % fvtool(emg_bpf,'Analysis','freq','Fs',emg_Fs);  % Essential to visualize frequency response properly.
    for i = 1:size(raw_emg,1)
        BPFred_emg(i,:) = filtfilt(num_bpf,den_bpf,double(raw_emg(i,:)));
        %BPFred_emg(i,:) = filtfilt(emg_bpf.sos.sosMatrix,emg_bpf.sos.ScaleValues,double(raw_emg(i,:)));
        %notch_emg(i,:) = filtfilt(emg_notch.sos.sosMatrix,emg_notch.sos.ScaleValues,double(raw_emg(i,:)));
    end
    
    Diff_emg(1,:) = BPFred_emg(1,:) - BPFred_emg(3,:);
    Diff_emg(2,:) = BPFred_emg(2,:) - BPFred_emg(4,:);
    
    emgt = 0:1/emg_Fs:(size(EEG.data,2)-1)/emg_Fs;
    figure; plot(emgt, Diff_emg(1,:));
    hold on; plot(emgt, Diff_emg(2,:),'r');
    
       
%     % Apply notch filter
%     [z p k] = butter(4, [4 7]./(emg_Fs/2), 'stop'); % 10th order filter
%     [sos,g]=zp2sos(z,p,k); % Convert to 2nd order sections form
%     notchf=dfilt.df2sos(sos,g); % Create filter object
%     %fvtool(notchf,'Analysis','freq','Fs',emg_Fs);
%     for i = 1:size(raw_emg,1)
%         raw_emg(i,:) = filtfilt(notchf.sos.sosMatrix,notchf.sos.ScaleValues,raw_emg(i,:));
%     end
      
    % ****STEP 2: Rectify & Lowpass Filter EMG Envelop, 2.2 Hz, 4th order Butterworth
    emglpfc = 1;       % Cutoff frequency = 2.2 Hz
    [num_lpf,den_lpf] = butter(4,(emglpfc/(emg_Fs/2)));
    %emg_lpf = dfilt.df2(num_lpf,den_lpf);       % Create filter object. Not essential
%   fvtool(emg_lpf,'Analysis','freq','Fs',emg_Fs);  % Essential to visualize frequency response properly.
    EMG_envelop = abs(Diff_emg);
    for k = 1:size(EMG_envelop,1)
        EMG_envelop(k,:) = filtfilt(num_lpf,den_lpf,EMG_envelop(k,:));
        %EMG_envelop(k,:) = filtfilt(emg_lpf.sos.sosMatrix,emg_lpf.sos.ScaleValues,EMG_envelop(k,:));
    end
    
     %**** STEP3: Calculate RMS using 500ms sliding window
   EMG_rms(1,:) = sqrt(smooth(Diff_emg(1,:).^2,150));
   EMG_rms(2,:) = sqrt(smooth(Diff_emg(2,:).^2,150));
   for k = 1:size(EMG_rms,1)
        EMG_rms(k,:) = filtfilt(num_lpf,den_lpf,EMG_rms(k,:));
        %EMG_envelop(k,:) = filtfilt(emg_lpf.sos.sosMatrix,emg_lpf.sos.ScaleValues,EMG_envelop(k,:));
   end
    
    figure; 
    hold on;plot(emgt,EMG_envelop(1,:),'b');   
    hold on; plot(emgt,EMG_envelop(2,:),'r');
    hold on;plot(emgt,EMG_rms(1,:),'k');   
    hold on; plot(emgt,EMG_rms(2,:),'g');
    
%     EEG.data(1,:) = BPFred_emg(1,:);
%     EEG.data(2,:) = BPFred_emg(2,:);
    EEG.data(1,:) = EMG_envelop(1,:);
    EEG.data(2,:) = EMG_envelop(2,:); 
    EEG.data(3,:) = Diff_emg(1,:);
    EEG.data(4,:) = Diff_emg(2,:);
    
    % **** STEP 5A: Remove extra stimulus triggers 'S  2'
    % Deletes 'S  2' trigger for trials which were aborted
    deleted_rest_trig_latency = [];
    for j=1:length(EEG.event)-1
        if (strcmp(EEG.event(j).type,rest_trig_label))
            if(strcmp(EEG.event(j+1).type,rest_trig_label))
                EEG.event(j).type = 'DEL';
                deleted_rest_trig_latency = [deleted_rest_trig_latency; EEG.event(j).latency/emg_Fs];                               
            end
        end
    end     
    EEG = eeg_checkset( EEG );
    
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_preprocessed.set'],'filepath',folder_path);    
    
% %  Calculate sliding and nonoverlapping window estimate of AR parameters
%     k = 1;
%     sig_samples  = 150; % 300ms window
%     ar_coeff = [];
%     while k < length(Diff_emg)
%         x_n = Diff_emg(1,k:k+sig_samples-1);
%         k = k+sig_samples-1; 
%         [ar_a,ar_v,ar_k] = aryule(x_n,4);
%         ar_coeff = [ar_coeff repmat(ar_a',1, sig_samples)];  
%     end
%     
    if extract_emg_epochs == 1      
        % Extract emg move epochs
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_preprocessed.set'], folder_path); 
        EEG = pop_epoch( EEG, {  move_trig_label  }, [-2.5 3], 'newname', 'move_epochs', 'epochinfo', 'yes');
        EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_emg_move_epochs'];
        EEG = eeg_checkset( EEG );
        
        if ~isempty(remove_corrupted_epochs)
            reject_epochs = zeros(1,size(EEG.epoch,2));
            reject_epochs(remove_corrupted_epochs) = 1;
            EEG = pop_rejepoch(EEG,reject_epochs,1);
        end

        % Save dataset
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_move_epochs.set'],'filepath',folder_path);    

        %Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        eeglab redraw;
               
        Fs_eeg = EEG.srate;
        %emg_erp_time = round((EEG.xmin:1/Fs_eeg:EEG.xmax)*100)./100;
        emg_erp_time = (EEG.xmin:1/Fs_eeg:EEG.xmax);

        [no_channels,no_datapts,no_epochs] = size(EEG.data);
        emg_move_epochs = zeros([no_epochs,no_datapts,no_channels]);        
        for epoch_cnt = 1:no_epochs
            for channel_cnt = 1:no_channels
                emg_move_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt);
            end
        end
                
        % Extract emg rest epochs ----------------------- Added 11-4-2014
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_preprocessed.set'], folder_path); 
        EEG = pop_epoch( EEG, {  rest_trig_label  }, [-2.5 3], 'newname', 'rest_epochs', 'epochinfo', 'yes');
        EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_emg_rest_epochs'];
        EEG = eeg_checkset( EEG );
        
        if ~isempty(remove_corrupted_epochs)
            reject_epochs = zeros(1,size(EEG.epoch,2));
            reject_epochs(remove_corrupted_epochs) = 1;
            EEG = pop_rejepoch(EEG,reject_epochs,1);
        end

        % Save dataset
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_rest_epochs.set'],'filepath',folder_path);    

        %Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        eeglab redraw;
        
        [no_channels,no_datapts,no_epochs] = size(EEG.data);
        emg_rest_epochs = zeros([no_epochs,no_datapts,no_channels]);        
        for epoch_cnt = 1:no_epochs
            for channel_cnt = 1:no_channels
                emg_rest_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt);
            end
        end
        
        emg_epochs.emg_move_epochs = emg_move_epochs;
        emg_epochs.emg_rest_epochs = emg_rest_epochs;
        emg_epochs.Fs_emg = EEG.srate;
        emg_epochs.xmin = EEG.xmin;
        emg_epochs.xmax = EEG.xmax;
        emg_epochs.emg_erp_time = emg_erp_time;
        
        filename10 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_emg_epochs.mat'];
        save(filename10,'emg_epochs');
        
%         figure; 
%         subplot(2,1,1); grid on; hold on;
%         plot(emg_erp_time,emg_move_epochs(47,:,1),'b','LineWidth',2)
%         hold on; plot(emg_erp_time,emg_move_epochs(47,:,2),'r','LineWidth',2)
%         title('Target: Up');
%         legend('Bicep','Tricep');
%         
%         subplot(2,1,2); grid on; hold on;
%         plot(emg_erp_time,emg_move_epochs(48,:,1),'b','LineWidth',2)
%         hold on; plot(emg_erp_time,emg_move_epochs(48,:,2),'r','LineWidth',2)
%         title('Target: Down');
          
%% Calulate Movement Onset using EMG
%           % 1. Apply baseline correction
%           emg_baseline = emg_epochs.emg_move_epochs(:,find(emg_epochs.emg_erp_time == -2.5):...
%         find(emg_epochs.emg_erp_time == -2.25),1:2);
%           
%           bc_emg_epochs(:,:,1) = emg_epochs.emg_move_epochs(:,:,1) -...
%         repmat(mean(emg_baseline(:,:,1),2),1,size(emg_epochs.emg_move_epochs,2));
%       
%           bc_emg_epochs(:,:,2) = emg_epochs.emg_move_epochs(:,:,2) -...
%         repmat(mean(emg_baseline(:,:,2),2),1,size(emg_epochs.emg_move_epochs,2));  
%           
%         % 2. 
%           % Up Targets
%           up_emg_epochs =  bc_emg_epochs(find(Velocity_Epoch(:,end)==3)-1,:,:);
% %           figure; plot(emg_epochs.emg_erp_time,up_emg_epochs(:,:,1)','b')
% %           hold on; plot(emg_epochs.emg_erp_time,up_emg_epochs(:,:,2)','r')
%           up_emg_max   = max(up_emg_epochs(:,:,1));
%           up_emg_thr = 0.1*(median(up_emg_max))
%           for m = 1:size(up_emg_epochs,1)
%             %up_emg_thr      = 0.05*up_emg_max(m);
%             up_onset_time(m) = emg_epochs.emg_erp_time(find(up_emg_epochs(m,:,1) >= up_emg_thr,1,'first'));
%           end
%           
%        
%           % Down Targets
%           down_emg_epochs = bc_emg_epochs(find(Velocity_Epoch(:,end)==1)-1,:,:);
% %           figure; plot(emg_epochs.emg_erp_time,down_emg_epochs(:,:,1)','b')
% %           hold on; plot(emg_epochs.emg_erp_time,down_emg_epochs(:,:,2)','r')
%           down_emg_max   = max(down_emg_epochs(:,:,2));
%           down_emg_thr = 0.1*(median(down_emg_max))
%           for m = 1:size(down_emg_epochs,1)
%              down_onset_time(m) = emg_epochs.emg_erp_time(find(down_emg_epochs(m,:,2) >= down_emg_thr,1,'first'));
%           end
%             
%           figure; subplot(2,1,1);
%           hist(up_onset_time);
%           subplot(2,1,2);
%           hist(down_onset_time);
%           median([up_onset_time down_onset_time(down_onset_time<0)])
%           
% % %           t1 = find(Velocity_Epoch(1,:) == -2.5);
% % %           t2 = find(Velocity_Epoch(1,:) == 1.00);
% % %           up_Velocity_epochs = Velocity_Epoch((Velocity_Epoch(:,end) == 3),t1:t2);
% % %           vel_epochs_time = Velocity_Epoch(1,t1:t2);
% % %           up_vel_max = max(up_Velocity_epochs,[],2);
% % %           
% % %           for m = 1:size(up_Velocity_epochs,1)
% % %             %up_emg_thr      = 0.05*up_emg_max(m);
% % %             up_vel_thr = 0.768;
% % %             up_vel_onset_time(m) = vel_epochs_time(find(up_Velocity_epochs(m,:) >= up_vel_thr,1,'first'));
% % %           end
          
    end
end
%% Processing of MAHI-Exo Kinematics
% ********** Matrix Columns ****************
%        ----Positions (rad, rad, m, m, m)-----    -------Velocities(rad/s and m/s)-----     ---------Torques (Nm)---------- 	 	-------EEG Related------ 	 
%Time(s) Elbow  Forearm  Wrist_1 Wrist_2 Wrist_3  Elbow  Forearm  Wrist_1 Wrist_2 Wrist_3   Elbow  Forearm Wrist_1 Wrist_2 Wrist_3  Trig1 Trig2 Trig_Mov FAKE Target  FAKE   Count
%  1       2       3        4       5      6        7       8        9      10      11       12      13      14     15       16      17    18     19      20    21      22    23
% *************
if label_events == 1
    
    if size(kin_blocks,1) == 2
        kin_sess_num = ['1';'2'];
    else
        kin_sess_num = Sess_num;
    end
    
   Kinematic_trajectory = [];
   count = 1;
   velocity_epoch_dur = [-4 6];
   Fs_kin = 200;
   Position_Epoch = [];
   Velocity_Epoch = [];
   Torque_Epoch = [];
   Velocity_Epoch(1,:) = [velocity_epoch_dur(1):1/Fs_kin:velocity_epoch_dur(2) 0];
   Position_Epoch(1,:) = [velocity_epoch_dur(1):1/Fs_kin:velocity_epoch_dur(2) 0];
   Torque_Epoch(1,:) = [velocity_epoch_dur(1):1/Fs_kin:velocity_epoch_dur(2)];
   
                        
    for ses_n = 1:length(kin_sess_num)
        Sess_num = kin_sess_num(ses_n);
        kin_first_block = kin_blocks(ses_n,1);
        kin_last_block = kin_blocks(ses_n,4);
    
    
        for kin_block_num = kin_first_block:kin_last_block
                if kin_block_num == 0
                    continue;
                else
                    %Redefine folder path
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change8
                    %kinematics_raw = importdata([Subject_name '_kinematics.mat']); % Raw data sampled at 200 Hz
                    kinematics_raw = dlmread([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num)...
                    '_block' num2str(kin_block_num) '_kinematics.txt'],'\t',14,1); % Raw data sampled at 200 Hz
                    Fs_kin_raw = 1000;  % For MAHI, sampling at 1kHz

                    % Correct data before initialization triggers. i.e. data segment before
                    % simultaneous stimulus and response triggers are received.
                    if find(kinematics_raw(:,17)==0,1,'first') == find(kinematics_raw(:,18)==0,1,'first')
                       kinematics_raw(1:find(kinematics_raw(:,17)==0,1,'first'),:) = [];
                       trig_correction = find(kinematics_raw(:,17)==5,1,'first');
                       kinematics_raw(1:trig_correction,17:18) = 5;
                    else
                        error('No Initialization Triggers found. Check kinematics data');
                    end

                    % Set zero time
                    t0_kin = kinematics_raw(1,1);
                    kinematics_raw(:,1) = kinematics_raw(:,1)-t0_kin;

                    % Downsample to 200 Hz; better than resample() 
                    kinematics = downsample(kinematics_raw,5); % Decimation by 1/5
                    kinematics = [kinematics; ones(1000,size(kinematics,2))]; % Padding with ones to compensate for missing data
                    Fs_kin = 200;      % Frequency

                    % Low Pass Filter Position and Velocity
                    position_f = []; velocity_f = [];
                    [fnum,fden] = butter(4,4/(Fs_kin/2),'low');
                    %freqz(fnum,fden,128,Fs_kin);
                    position_f(:,1) = (180/pi).*filtfilt(fnum,fden,kinematics(:,2));      % Elbow position (deg)
                    %position_f(:,2) = filtfilt(fnum,fden,kinematics(:,3));
                    velocity_f(:,1) = (180/pi).*filtfilt(fnum,fden,kinematics(:,7));     % Elbow velocity (deg/s)
                    %velocity_f(:,1) = (180/pi).*sgolayfilt(kinematics(:,7),3,51);
                    %velocity_f(:,2) = filtfilt(fnum,fden,kinematics(:,5));
                    %tan_velocity = sqrt(velocity_f(:,1).^2 + velocity_f(:,2).^2);  %InMotion
                    tan_velocity = velocity_f(:,1);
                    torque_f =  kinematics(:,12);

                    %****** Extract unique triggers and modify External Triggers to include only first sample instant
                    % figure; plot(kinematics(:,1),kinematics(:,8));
                    temp_response_trig = kinematics(:,18);
                    temp_stimulus_trig = kinematics(:,17);
                    temp_move_trig = kinematics(:,19);
                    [kin_response_trig_index,kinematics(:,18)] = ExtractUniqueTriggers(temp_response_trig,1,0); 
                    [kin_stimulus_trig_index,kinematics(:,17)] = ExtractUniqueTriggers(temp_stimulus_trig,1,0); 
                    [kin_move_trig_index,kinematics(:,19)] = ExtractUniqueTriggers(temp_move_trig,1,0); 

                    [intersect_val, intersect_stimulus,intersect_response] = intersect(kin_stimulus_trig_index,kin_response_trig_index);
                    kin_stimulus_trig_index(intersect_stimulus) = [];
                    kin_response_trig_index(intersect_response) = [];
                    
                    %% Extract position, velocity, torque epochs , direction                                                     
                         for k = 1:length(kin_move_trig_index)
                            count = count + 1;
                            limit1 = kin_move_trig_index(k);
                            limit2 = kin_response_trig_index(k);
                            % Includes trial_no, position, velocity, torque and target information
                            Kinematic_trajectory = [Kinematic_trajectory; ...
                                [count.*ones(limit2-limit1+1,1) ...
                                position_f(limit1:limit2) ...
                                tan_velocity(limit1:limit2) ...
                                torque_f(limit1:limit2) ...
                                kinematics(kin_move_trig_index(k),21).*ones(limit2-limit1+1,1)]];      
                        end
                         
                    
                        % -OR- Extract velocity & position epochs around movement onset - Later we can check
                        % with EEG signals
                        % Velocity_Epoch/Position_Epoch = [time        0; 
                        %                                  velocities  target]
                        % Targets: 3 - Up; 1 - Down; 2 - Center; 0,4 - In between trials
                        % Extract Torque epochs similar to Velocity Epochs
                        % Torque_Epoch = [time; 
                        %                 torques_for_each_trial];

                        
                        for k = 1:length(kin_move_trig_index)
                            limit1 = kin_move_trig_index(k) - round(abs(velocity_epoch_dur(1))*Fs_kin);
                            limit2 = kin_move_trig_index(k) + round(abs(velocity_epoch_dur(2))*Fs_kin);
                            %tan_velocity = [tan_velocity; zeros(100,1)];
                            Velocity_Epoch = [Velocity_Epoch; [tan_velocity(limit1:limit2)' kinematics(kin_move_trig_index(k),21)]];  % Includes velocity and target information
                            Position_Epoch = [Position_Epoch; [position_f(limit1:limit2)' kinematics(kin_move_trig_index(k),21)]];  
                            Torque_Epoch = [Torque_Epoch; torque_f(limit1:limit2)'];
                        end   
                end
        end % end first block, last block loop
    end
   
                        if ~isempty(remove_corrupted_epochs)
                            Velocity_Epoch(remove_corrupted_epochs+1,:) = [];       % +1 is added since 1st row is time duration for velocity
                            Position_Epoch(remove_corrupted_epochs+1,:) = [];
                            Torque_Epoch(remove_corrupted_epochs+1,:) = [];
                        end

                        kinematics_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_velocity_epochs.mat'];
                        save(kinematics_filename,'Velocity_Epoch');
                        kinematics_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_position_epochs.mat'];
                        save(kinematics_filename,'Position_Epoch');
                        kinematics_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_torque_epochs.mat'];
                        save(kinematics_filename,'Torque_Epoch');
                        kinematics_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_kinematic_trajectory.mat'];
                        save(kinematics_filename,'Kinematic_trajectory');
                      
%% Plotting 
% %                         baseline_index1 = find(Torque_Epoch(1,:) == -2.5);
% %                         baseline_index2 = find(Torque_Epoch(1,:) == -2.0);
% %                         trials_from = 1;
% %                         trials_to = 80; %size(emg_epochs.emg_move_epochs,1);
% %                         figure;  
% %                         T_plot = tight_subplot(4,1,[0.07 0.01],[0.1 0.1],[0.1 0.1]);
% % 
% %                         %subplot(3,1,1); 
% %                 %         axes(T_plot(1));
% %                 %         hold on; grid on;
% %                 %         for k = 1:(size(Position_Epoch,1)-1)
% %                 %             plot(Position_Epoch(1,1:(end-1)),Position_Epoch(k+1,1:(end-1)));
% %                 % %                - mean(Position_Epoch(k+1,baseline_index1:baseline_index2)),'b');
% %                 %         end
% %                 %         %xlabel('Time (sec.)')
% %                 %         xlim([-2.5 3]);
% %                 %         ylim([-60 0])
% %                 %         ylabel('Elbow Position (deg)','FontSize',12)
% %                 %         set(gca,'XTick',[-2 -1 0 1 2 3]);
% %                 %         set(gca,'XTickLabel','');
% %                 %         v = axis;
% %                 %         line([0 0],[v(3) v(4)],'Color','k');
% % 
% %                         %subplot(3,1,2);
% %                         axes(T_plot(1));
% %                         hold on; grid on;
% %                         for k = trials_from:trials_to   %1:(size(Velocity_Epoch,1)-1)
% %                             plot(Velocity_Epoch(1,1:(end-1)),Velocity_Epoch(k+1,1:(end-1)));
% %                 %                - mean(Velocity_Epoch(k+1,baseline_index1:baseline_index2)),'b');
% %                         end
% %                         %xlabel('Time (sec.)')
% %                         xlim([-2.5 3]);
% %                         ylim([-10 10])
% %                         ylimits = ylim;
% %                         title('Velocity (deg/s)','FontSize',12)
% %                         %lh = legend('Velocity \n (deg/s)','Location','NorthWest');
% %                         %set(lh,'Box', 'off','Color','none');
% %                         set(gca,'XTick',[-2 -1 0 1 2 3]);
% %                         set(gca,'XTickLabel','');
% %                         set(gca,'YTick',[ylimits(1) 0 ylimits(2)]);
% %                         set(gca,'YTickLabel',{num2str(ylimits(1)) '0' num2str(ylimits(2))},'FontSize',11);
% %                         v = axis;
% %                         line([0 0],[v(3) v(4)],'Color','k');    
% % 
% %                         %subplot(3,1,3);
% %                         axes(T_plot(2))
% %                         hold on; grid on;
% %                         for k = trials_from:trials_to    %1:(size(Velocity_Epoch,1)-1)
% %                             % No baseline correction
% %                             plot(Torque_Epoch(1,:),Torque_Epoch(k+1,:));%- mean(Torque_Epoch(k+1,baseline_index1:baseline_index2)),'b');
% %                         end
% %                         ylim([-0.2 0.2]);
% %                         ylimits = ylim;
% %                         xlim([-2.5 3]);
% %                         title('Torque (Nm)','FontSize',12)
% %                 %         lh = legend('Torque','Location','NorthWest');
% %                 %         set(lh,'Box', 'off','Color','none');
% %                         set(gca,'XTick',[-2 -1 0 1 2 3]);
% %                         set(gca,'XTickLabel','');
% %                         set(gca,'YTick',[ylimits(1) 0 ylimits(2)]);
% %                         set(gca,'YTickLabel',{num2str(ylimits(1)) '0' num2str(ylimits(2))},'FontSize',11);
% %                         v = axis;
% %                         line([0 0],[v(3) v(4)],'Color','k');    
% % 
% % 
% %                         emg_baseline_index1 = round(find(emg_epochs.emg_erp_time == -2.5,1,'first'));
% %                         emg_baseline_index2 = round(find(emg_epochs.emg_erp_time == -2.25,1,'first'));
% % 
% %                         axes(T_plot(3))
% %                         hold on; grid on;
% %                         for k = trials_from:trials_to     %1:size(emg_epochs.emg_move_epochs,1)
% %                             plot(emg_epochs.emg_erp_time,emg_epochs.emg_move_epochs(k,:,1)...
% %                                 - mean(emg_epochs.emg_move_epochs(k,emg_baseline_index1:emg_baseline_index2,1)),'b');
% %                         end
% %                         ylim([0 60]);
% %                         ylimits = ylim;
% %                         xlim([-2.5 3]);
% %                         title('Biceps Activity (mV)','FontSize',12)
% %                 %         lh = legend('Biceps Activity','Location','NorthWest');
% %                 %         set(lh,'Box', 'off','Color','none');
% %                         set(gca,'XTick',[-2 -1 0 1 2 3]);
% %                         set(gca,'XTickLabel','');
% %                         set(gca,'XTickLabel','');
% %                         set(gca,'YTick',[0 ylimits(2)]);
% %                         set(gca,'YTickLabel',{'0' num2str(ylimits(2))},'FontSize',11);
% %                         v = axis;
% %                         line([0 0],[v(3) v(4)],'Color','k');    
% % 
% % 
% %                         axes(T_plot(4))
% %                         hold on; grid on;
% %                         for k = trials_from:trials_to    %size(emg_epochs.emg_move_epochs,1)
% %                             plot(emg_epochs.emg_erp_time,emg_epochs.emg_move_epochs(k,:,2)...
% %                                 - mean(emg_epochs.emg_move_epochs(k,emg_baseline_index1:emg_baseline_index2,2)),'b');
% %                         end
% %                         ylim([0 60]);
% %                         ylimits = ylim;
% %                         xlim([-2.5 3]);
% %                         title('Triceps Activity (mV)','FontSize',12)
% %                 %         lh = legend('Triceps Activity','Location','NorthWest');
% %                 %         set(lh,'Box', 'off','Color','none');
% %                         set(gca,'XTick',[-2 -1 0 1 2 3]);
% %                         set(gca,'XTickLabel',{'-2' '-1' '0' '1' '2' '3'});
% %                         set(gca,'YTick',[0 ylimits(2)]);
% %                         set(gca,'YTickLabel',{'0' num2str(ylimits(2))},'FontSize',11);
% %                         v = axis;
% %                         line([0 0],[v(3) v(4)],'Color','k');    
% % 
% % 
% %                         mtit('Subject H3, Backdrive Mode', 'FontSize',12,'xoff',0,'yoff',0.05);
% %                         xlabel('Time (sec.)','FontSize',12)


%export_fig 'MR_ses1_cond4_vel_torq_emg' '-png' '-transparent'

end

if use_kinematics_old_code == 1
%% Extract trigger latencies from EEG signals
% Ensure SB_preprocessed.set/SB_standardized.set is loaded into EEGLAB

if ~isempty(Cond_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'], folder_path);
elseif ~isempty(Block_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'], folder_path); 
else
    EEG = pop_loadset( [Subject_name '_preprocessed.set'], folder_path); 
end

Fs_eeg = EEG.srate;
eeg_response_trig = [];     % Trigger event when robot moves
eeg_stimulus_trig = [];     % Trigger event when target appears
eeg_move_trig = [];
initial_time_flag = 0;

% % freq = 1000;
% % block_start = []; block_stop = [];
% % for j=1:length(EEG.event)
% %     if (strcmp(EEG.event(j).type,'S 10'))
% %         block_start = [block_start; EEG.event(j).latency - freq];
% %     elseif (strcmp(EEG.event(j).type,'S 26'))
% %         block_stop = [block_stop; EEG.event(j).latency + freq];
% %     end
% % end
% % disp(block_start); disp(block_stop);


for j=1:length(EEG.event)
    if (strcmp(EEG.event(j).type,'S  8'))
        eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
    elseif (strcmp(EEG.event(j).type,move_trig_label))
        eeg_move_trig = [eeg_move_trig; EEG.event(j).latency/Fs_eeg];
    elseif (strcmp(EEG.event(j).type,rest_trig_label))
        eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
    elseif (strcmp(EEG.event(j).type,'S 10'))
        if initial_time_flag == 0
            initial_time_flag = 1;
            stimulus_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
            response_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
        else
            eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
            eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
        end
    end
end
eeg_response_trig = eeg_response_trig/Fs_eeg;       % Convert to seconds.
eeg_stimulus_trig = eeg_stimulus_trig/Fs_eeg;
eeg_stimulus_trig = eeg_stimulus_trig - stimulus_time_correction;   % Apply correction, make t = 0
eeg_response_trig = eeg_response_trig - response_time_correction;   % Apply correction, make t = 0  

% Round triggers upto two decimal places
eeg_stimulus_trig = (ceil(eeg_stimulus_trig.*100))/100;
eeg_response_trig = (ceil(eeg_response_trig.*100))/100;
kinematics(:,1) = (round(kinematics(:,1).*1000))/1000;

valid_cnt = 0;          % Number of valid trials 
rest_cnt = 0;
tan_velocity_epoch = [];
Move_onset_indices = zeros(length(eeg_response_trig),1);
Move_onset_trig = [[] []];
Target_cue_trig = [[] []];
response_ind = 1;
for trig_cnt = 1:length(eeg_stimulus_trig)-1
%    if trig_cnt + 1 <= length(eeg_stimulus_trig)       
        if (eeg_response_trig(response_ind) > eeg_stimulus_trig(trig_cnt)) && (eeg_response_trig(response_ind) < eeg_stimulus_trig(trig_cnt+1))

            limit1 = find(kinematics(:,1)==eeg_stimulus_trig(trig_cnt))+200;                % Required - Wait 1 sec after end of previous movement
            limit2 = find(kinematics(:,1)==eeg_response_trig(response_ind));            

            tan_velocity_epoch = tan_velocity(limit1:limit2);
            tan_velocity_max   = max(tan_velocity_epoch);
            vel_threshold      = 0.05*tan_velocity_max;
            Move_onset_indices(trig_cnt) = find(tan_velocity_epoch >= vel_threshold,1,'first') + limit1 -1; %Important to add limit1
            if((Move_onset_indices(trig_cnt) - limit1) >= 1.0*Fs_eeg)     % Wait atleast 2 secs after end of previous movement.
                valid_cnt = valid_cnt + 1;
                Move_onset_trig = [Move_onset_trig; [100 kinematics(Move_onset_indices(trig_cnt),1)]];   % '100' is for appending event to EEG dataset
                Target_cue_trig = [Target_cue_trig; [200 kinematics(limit1 - 100,1)]];   % '200' is for appending event to EEG dataset
            end
        else
            rest_cnt = rest_cnt + 1;
            response_ind = response_ind - 1;
        end
        response_ind = response_ind + 1;
%    end
end

% Plot Stimulus & Movement onset triggers on tangential velocity
lastpt = length(kinematics(:,1));
figure; hold on;
plot(kinematics(1:lastpt,1),tan_velocity(1:lastpt),'k','LineWidth',2);
plot(kinematics(1:lastpt,1),zscore(position_f(1:lastpt))./100,'r','LineWidth',2);
myaxis = axis;
for plot_ind1 = 1:length(eeg_stimulus_trig);
    line([eeg_stimulus_trig(plot_ind1), eeg_stimulus_trig(plot_ind1)],[myaxis(3), myaxis(4)],'Color','k','LineWidth',2,'LineStyle','--');
    hold on;
end

% for plot_ind6 = 1:length(eeg_response_trig);
%     line([eeg_response_trig(plot_ind6), eeg_response_trig(plot_ind6)],[myaxis(3), myaxis(4)],'Color','b','LineWidth',2);
%     hold on;
% end

% for plot_ind2 = 1:length(Move_onset_trig);
%     line([Move_onset_trig(plot_ind2,2), Move_onset_trig(plot_ind2,2)],[myaxis(3), myaxis(4)],'Color','r','LineWidth',2);
%     hold on;
% end

for plot_ind7 = 1:length(kin_move_trig_index);
    line([kinematics(kin_move_trig_index,1), kinematics(kin_move_trig_index,1)],[myaxis(3), myaxis(4)],'Color','b','LineWidth',2);
    hold on;
end
hold off;
%% Generate Raster Plot  
%     raster_colors = ['g','r','b','k','y','c','m','g','r','b','k','r','b','k','g','r','b','k','y','c','m'];
%     raster_Fs = Fs_kin;
%     first_trig = 10;
%     last_trig  = 15;
%     raster_lim1 = kin_stimulus_trig_index(first_trig)-Fs_kin;   % Take previous 1 second
%     raster_lim2 = kin_stimulus_trig_index(last_trig)+Fs_kin;   % Take next 1 second
%     raster_time = kinematics(raster_lim1:raster_lim2,1);
%     eeg_raster_lim1 = round(stimulus_time_correction*100)+raster_lim1;
%     eeg_raster_lim2 = eeg_raster_lim1 + (raster_lim2-raster_lim1);  %Matrix dimension mismatch
% 
%     % Plot selected EEG channels and kinematics
%     raster_data = zeros(length(position_f(raster_lim1:raster_lim2,1)),length(Channels_nos));
%     for chan_index = 1:length(Channels_nos)
%         raster_data(:,chan_index) = EEG.data(Channels_nos(chan_index),eeg_raster_lim1:eeg_raster_lim2)';
%     end
%     raster_data = [ raster_data position_f(raster_lim1:raster_lim2,1) position_f(raster_lim1:raster_lim2,2) ...
%         tan_velocity(raster_lim1:raster_lim2) ];
% 
%     % Standardization/Normalization of Signals; 
%     % http://en.wikipedia.org/wiki/Standard_score
%     % Standard score or Z-score is the number of standard deviations an
%     % obervation is above/below the mean. 
%     % z-score = (X - mu)/sigma;
% 
%     raster_zscore = zscore(raster_data);
% 
%     % Plot the rasters; Adjust parameters for plot
%     [raster_row,raster_col] = size(raster_zscore);
%     add_offset = 5;
%     raster_ylim1 = 0;
%     raster_ylim2 = (raster_col+1)*add_offset;
%     figure;
%     for raster_index = 1:raster_col;
%         raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
%         plot(raster_time,raster_zscore(:,raster_index),raster_colors(raster_index));
%         hold on;
%     end
%     for plot_ind1 = first_trig:last_trig;          % Plot & label the stimulus triggers
%         line([kin_stimulus_trig_index(plot_ind1)/raster_Fs, kin_stimulus_trig_index(plot_ind1)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','r','LineWidth',1.5,'LineStyle','-');
%         text(kin_stimulus_trig_index(plot_ind1)/raster_Fs,raster_ylim2,'New Target','Rotation',60,'FontSize',9);
%         hold on;
%     end
%     % % for plot_ind2 = first_trig:(last_trig-1);          % Plot & label the response triggers
%     % %     line([kin_response_trig_index(plot_ind2)/raster_Fs, kin_response_trig_index(plot_ind2)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','r','LineWidth',1);
%     % %     text(kin_response_trig_index(plot_ind2)/raster_Fs,raster_ylim2,'Move','Rotation',60,'FontSize',8);
%     % %     hold on;
%     % % end
%     for plot_ind2 = first_trig+1:last_trig;          % Plot & label the response triggers
%         line([kin_response_trig_index(plot_ind2)/raster_Fs, kin_response_trig_index(plot_ind2)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','b','LineWidth',1.5,'LineStyle','-');
%         text(kin_response_trig_index(plot_ind2)/raster_Fs,raster_ylim2+1,'Hit','Rotation',60,'FontSize',10);
%         %text((kin_response_trig_index(plot_ind2)/raster_Fs)+0.25,raster_ylim2,'Reached','Rotation',60,'FontSize',8);
%         hold on;
%     end
% 
%     for plot_ind3 = 1:length(Move_onset_trig)
%         if ((Move_onset_trig(plot_ind3,2) >= raster_lim1/raster_Fs) && (Move_onset_trig(plot_ind3,2) <= raster_lim2/raster_Fs))
%             line([Move_onset_trig(plot_ind3,2), Move_onset_trig(plot_ind3,2)],[raster_ylim1,raster_ylim2],'Color','k','LineWidth',2,'LineStyle','--');
%             text(Move_onset_trig(plot_ind3,2),raster_ylim2,'Move Onset','Rotation',60,'FontSize',9);
%             %text(Move_onset_trig(plot_ind3,2)+0.25,raster_ylim2,'Onset','Rotation',60,'FontSize',8);
%         end
%     end
%     axis([raster_lim1/raster_Fs raster_lim2/raster_Fs raster_ylim1 raster_ylim2]);
%     set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85]);
%     set(gca,'YTickLabel',{'F1','Fz','F2','FC1','FCz','FC2','C1','Cz','C2','CP1','CPz','CP2','C3','C4','Pos X ','Pos Y','Tang Velocity'},'FontSize',8);
%     %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
%     %set(gca,'XTickLabel',{'Target','Movement'});
%     xlabel('Time (sec.)','FontSize',10);
%     hold off;
% 
%% Add Stimulus time correction for EEG latencies
    Move_onset_trig(:,2) = Move_onset_trig(:,2) + stimulus_time_correction;
    Target_cue_trig(:,2) = Target_cue_trig(:,2) + stimulus_time_correction;
    %events_filename = ['F:\\Nikunj_Data\\InMotion_Experiment_Data\\Subject_' Subject_name '\\Movement_events.txt'];
    
    if ~isempty(Cond_num)
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num)  '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_Movement_events.txt'];
    elseif ~isempty(Block_num)
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_Movement_events.txt'];
    else
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_Movement_events.txt'];
    end
    disp(events_filename);
    dlmwrite(events_filename,Move_onset_trig,'delimiter','\t','precision','%d %.4f');
    
    
    if ~isempty(Cond_num)
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num)  '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_Cue_events.txt'];
    elseif ~isempty(Block_num)
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_Cue_events.txt'];
    else
        events_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_Cue_events.txt'];
    end
    
    dlmwrite(events_filename,Target_cue_trig,'delimiter','\t','precision','%d %.4f')

    % Append events/triggers  
%    EEG = pop_importevent(EEG);
%    EEG = eeg_checkset(EEG);   
%     disp(events_filename);
%     EEG = pop_importevent(EEG);
%     EEG = eeg_checkset(EEG);
% % Save dataset
%     if ~isempty(Cond_num)
%     EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'],'filepath',folder_path);
%     elseif ~isempty(Block_num)
%     EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'],'filepath',folder_path);    
%     else
%     EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_preprocessed.set'],'filepath',folder_path);
%     end
%  
% % Update EEGLAB window
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%     eeglab redraw;
%% Load targets file. Prepare targets_full array
% % Add the center position for each target (N,S,E,W) position 
if include_target_events == 1
    
    targets = importdata('targets.txt');      
    targets_full = [];

    for target_cnt = 1:length(targets)
       switch targets(target_cnt)
           case 0
               targets_full = [targets_full;0];
           case 1
               targets_full = [targets_full;1;0];
           case 2
               targets_full = [targets_full;2;0];
           case 3
               targets_full = [targets_full;3;0];
           case 4
               targets_full = [targets_full;4;0];
           case 5
               targets_full = [targets_full;5];
           otherwise warning('Error in targets');
       end      
    end
    % Assign target numbers to events, 14-8-2013

    % 0 - Center
    % 1 - South
    % 2 - West
    % 3 - North
    % 4 - East
    % 5 - Rest

    target_cnt = 1;
    for event_cnt = 4:length(EEG.event)         % Ignore boundary + start triggers
        if strcmp(EEG.event(event_cnt).type,'S  2')
            if target_cnt <= length(targets_full);        
                switch targets_full(target_cnt)
                    case 0
                        EEG.event(event_cnt).type = 'center';
                        target_cnt = target_cnt + 1;
                    case 1
                        EEG.event(event_cnt).type = 'south';
                        target_cnt = target_cnt + 1;

                    case 2
                        EEG.event(event_cnt).type = 'west';
                        target_cnt = target_cnt + 1;

                    case 3
                        EEG.event(event_cnt).type = 'north';
                        target_cnt = target_cnt + 1;

                    case 4
                        EEG.event(event_cnt).type = 'east';
                        target_cnt = target_cnt + 1;

                    case 5
                        EEG.event(event_cnt).type = 'rest';
                        target_cnt = target_cnt + 1;

                    otherwise warning('Error in switch');
                end
            end
        end
    end
end
end
%% Extract move epochs and rest epochs
if extract_epochs == 1
    
    %% Extract move epochs
    if ~isempty(Cond_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'], folder_path);
    elseif ~isempty(Block_num)
        %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed_event_labels.set'], folder_path);
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'], folder_path);
    else
        %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed_event_labels.set'], folder_path);
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed.set'], folder_path);
    end
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;        
       
    EEG = pop_epoch( EEG, {  move_trig_label  }, move_epoch_dur, 'newname', 'move_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    
%     if lpfc <= 1
%         EEG = pop_resample( EEG, 10);
%     end
    if ~isempty(remove_corrupted_epochs)
        reject_epochs = zeros(1,size(EEG.epoch,2));
        reject_epochs(remove_corrupted_epochs) = 1;
        EEG = pop_rejepoch(EEG,reject_epochs,1);
    end

    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_move_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    if ~isempty(Cond_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_move_epochs.set'],'filepath',folder_path);    
    elseif ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_move_epochs.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_move_epochs.set'],'filepath',folder_path);
    end
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    %% Extract rest epochs
    %Subject_name = Subject_name_old;
    if ~isempty(Cond_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'], folder_path);
    elseif ~isempty(Block_num)
        %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed_event_labels.set'], folder_path);
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'], folder_path);
    else
        %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed_event_labels.set'], folder_path);
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed.set'], folder_path);
    end
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;        

    EEG = pop_epoch( EEG, {  rest_trig_label  }, rest_epoch_dur, 'newname', 'rest_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
%     if lpfc <= 1
%         EEG = pop_resample( EEG, 10);
%     end
%remove_corrupted_epochs = [81];

    if ~isempty(remove_corrupted_epochs)
            reject_epochs = zeros(1,size(EEG.epoch,2));
            reject_epochs(remove_corrupted_epochs) = 1;
            EEG = pop_rejepoch(EEG,reject_epochs,1);
    end
    
    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_rest_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    if ~isempty(Cond_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_rest_epochs.set'],'filepath',folder_path);
    elseif ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_rest_epochs.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_rest_epochs.set'],'filepath',folder_path);
    end
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    %% Extract EEG epochs for decoding trajectories - Dec 13, 2014
    if segment_data_for_decoding_kinematics == 1
            if ~isempty(Cond_num)
                EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_eeg_preprocessed.set'], folder_path);
            elseif ~isempty(Block_num)
                %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed_event_labels.set'], folder_path);
                EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed.set'], folder_path);
            else
                %EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed_event_labels.set'], folder_path);
                EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_preprocessed.set'], folder_path);
            end
            [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
            eeglab redraw;        

             Fs_eeg = EEG.srate;
             eeg_response_trig = [];     % Trigger event when robot moves
             eeg_stimulus_trig = [];     % Trigger event when target appears
             eeg_move_trig = [];
             %initial_time_flag = 0;


                for j=1:length(EEG.event)
                    if (strcmp(EEG.event(j).type,target_reached_trig_label))
                        eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
                    elseif (strcmp(EEG.event(j).type,move_trig_label))
                        eeg_move_trig = [eeg_move_trig; EEG.event(j).latency];
                    elseif (strcmp(EEG.event(j).type,rest_trig_label))
                        eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
                    elseif (strcmp(EEG.event(j).type,'S 10'))
                        % Do nothing
%                         if initial_time_flag == 0
%                             initial_time_flag = 1;
%                             stimulus_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
%                             response_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
%                         else
%                             eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
%                             eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
%                        end
                    end
                end
                %eeg_response_trig = eeg_response_trig/Fs_eeg;       % Convert to seconds.
                %eeg_stimulus_trig = eeg_stimulus_trig/Fs_eeg;
                %eeg_move_trig = eeg_move_trig/Fs_eeg;
                
%                 eeg_stimulus_trig = eeg_stimulus_trig - stimulus_time_correction;   % Apply correction, make t = 0
%                 eeg_response_trig = eeg_response_trig - response_time_correction;   % Apply correction, make t = 0  
            
                EEGsignal = [];  
                count =0;
                for k = 1:length(eeg_move_trig)
                    count = count + 1;
                    EEG_epoch = [];
                    for lags = 0:6
                            limit1 = round(eeg_move_trig(k)) - lags*10;
                            limit2 = round(eeg_response_trig(k)) - lags*10;            % 10 samples = 50 ms
                            EEG_epoch = [EEG_epoch EEG.data(:,limit1:limit2)'];
                    end
                    EEGsignal = [EEGsignal; [EEG_epoch count*ones(limit2-limit1+1,1)]];
                end
                
%                 for u = 1:160
%                     eeglengths(u) = length(find(EEGsignal(:,449) == u));
%                     kinlengths(u) = length(find(Kinematic_trajectory(:,1) == u));
%                 end
%                 diff = (kinlengths - eeglengths); 
%                 diff(find(diff  ~= 0))
             eegsignal_filename = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_EEGsignal.mat'];
             save(eegsignal_filename,'EEGsignal');

               

    end
end
%% Manipulate & Calculate ERPs for Rest Epochs
if manipulate_epochs == 1
    if ~isempty(Cond_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_rest_epochs.set'], folder_path);
    elseif ~isempty(Block_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_rest_epochs.set'], folder_path);
    else
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_rest_epochs.set'], folder_path);
    end
Fs_eeg = EEG.srate;
rest_erp_time = round((EEG.xmin:1/Fs_eeg:EEG.xmax)*100)./100;

[no_channels,no_datapts,no_epochs] = size(EEG.data);
rest_epochs = zeros([no_epochs,no_datapts,no_channels]); 
rest_std_channels = zeros(no_channels,no_datapts);
rest_avg_channels = zeros(no_channels,no_datapts);
rest_SE_channels =[]; 
for epoch_cnt = 1:no_epochs
    for channel_cnt = 1:no_channels
        %rest_sum_channels(channel_cnt,:) = rest_sum_channels(channel_cnt,:) + EEG.data(channel_cnt,:,epoch_cnt);
        rest_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt); 
        % Calculate mean value during rest
        rest_mean_baseline(channel_cnt,epoch_cnt) = mean(rest_epochs(epoch_cnt,find(rest_erp_time == (baseline_int(1))):find(rest_erp_time == (baseline_int(2))),channel_cnt));     
        
        %1. Subtract mean value of rest from rest trials
    if apply_baseline_correction == 1 
        rest_epochs(epoch_cnt,:,channel_cnt) = rest_epochs(epoch_cnt,:,channel_cnt)-rest_mean_baseline(channel_cnt,epoch_cnt);
    end
        rest_epochs_with_base_correct(epoch_cnt,:,channel_cnt) = rest_epochs(epoch_cnt,:,channel_cnt)-rest_mean_baseline(channel_cnt,epoch_cnt);
        %rest_epochs(epoch_cnt,:,channel_cnt) = log(abs(rest_epochs(epoch_cnt,:,channel_cnt)-rest_mean_baseline(channel_cnt,epoch_cnt)));
        
        %2. Divide by baseline
        %rest_epochs(epoch_cnt,:,channel_cnt) = rest_epochs(epoch_cnt,:,channel_cnt)./rest_mean_baseline(channel_cnt,epoch_cnt);
        %rest_epochs(epoch_cnt,:,channel_cnt) = log(abs(rest_epochs(epoch_cnt,:,channel_cnt)./rest_mean_baseline(channel_cnt,epoch_cnt)));
        
        %3. Standardize
    if apply_epoch_standardization == 1
        rest_epochs(epoch_cnt,:,channel_cnt) = zscore(rest_epochs(epoch_cnt,:,channel_cnt),0,2);
    end
        %rest_epochs(epoch_cnt,:,channel_cnt) = log(abs(zscore(rest_epochs(epoch_cnt,:,channel_cnt),0,2)));
        
        %4. Normalize
        %rest_epochs(epoch_cnt,:,channel_cnt) = rest_epochs(epoch_cnt,:,channel_cnt)./max(abs(rest_epochs(epoch_cnt,1:find(rest_erp_time == 0),channel_cnt)));
        
    end
end

% Determine t-statistics for 95% C.I.
%http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s
deg_freedom = no_epochs - 1;
t_value = tinv(1 - 0.05/2, deg_freedom);

for channel_cnt = 1:no_channels
    rest_avg_channels(channel_cnt,:) =  mean(rest_epochs_with_base_correct(:,:,channel_cnt));
    rest_std_channels(channel_cnt,:) =  std(rest_epochs_with_base_correct(:,:,channel_cnt));
    rest_SE_channels(channel_cnt,:) = t_value.*std(rest_epochs_with_base_correct(:,:,channel_cnt))/sqrt(no_epochs);
end
% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
% Calculate ERPs for Movement Epochs
if ~isempty(Cond_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_move_epochs.set'], folder_path);
elseif ~isempty(Block_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_move_epochs.set'], folder_path);
else
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_move_epochs.set'], folder_path);
end

Fs_eeg = EEG.srate;
move_erp_time = round((EEG.xmin:1/Fs_eeg:EEG.xmax)*100)./100;

[no_channels,no_datapts,no_epochs] = size(EEG.data);
move_epochs = zeros([no_epochs,no_datapts,no_channels]); 
move_avg_channels = zeros(no_channels,no_datapts);
move_std_channels = zeros(no_channels,no_datapts);
move_SE_channels = [];

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
        
        %2. Divide by baseline
        %move_epochs(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt)./move_mean_baseline(channel_cnt,epoch_cnt);
        %move_epochs(epoch_cnt,:,channel_cnt) = log(abs(move_epochs(epoch_cnt,:,channel_cnt)./move_mean_baseline(channel_cnt,epoch_cnt)));

        %3. Standardize
    if apply_epoch_standardization == 1
        move_epochs(epoch_cnt,:,channel_cnt) = zscore(move_epochs(epoch_cnt,:,channel_cnt),0,2);
    end
        %move_epochs(epoch_cnt,:,channel_cnt) = log(abs(zscore(move_epochs(epoch_cnt,:,channel_cnt),0,2)));
        
        %4. Normalize
        %move_epochs(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt)./max(abs(move_epochs(epoch_cnt,1:find(move_erp_time == 0),channel_cnt)));
    end
end

for channel_cnt = 1:no_channels
%     move_avg_channels(channel_cnt,:) =  mean(move_epochs(:,:,channel_cnt));
%     move_std_channels(channel_cnt,:) =  std(move_epochs(:,:,channel_cnt));
%     move_SE_channels(channel_cnt,:) = 1.96.*std(move_epochs(:,:,channel_cnt))/sqrt(no_epochs);
    move_avg_channels(channel_cnt,:) =  mean(move_epochs_with_base_correct(:,:,channel_cnt));
    move_std_channels(channel_cnt,:) =  std(move_epochs_with_base_correct(:,:,channel_cnt));
    move_SE_channels(channel_cnt,:) = t_value.*std(move_epochs_with_base_correct(:,:,channel_cnt))/sqrt(no_epochs);
end

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end

%separabiltiy_index;
%% Plot ERPs
if plot_ERPs == 1
    paper_font_size = 10;
    figure('Position',[1050 1300 3.5*116 2.5*116]); 
    %figure('units','normalized','outerposition',[0 0 1 1])
    T_plot = tight_subplot(3,5,[0.01 0.01],[0.15 0.01],[0.1 0.1]);
    hold on;
    plot_ind4 = 1;
    
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
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:),'r','LineWidth',2);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:)+ (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:) - (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         
%         jbfill(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),...
%            move_avg_channels(Channels_nos(ind4),:)- (move_SE_channels(Channels_nos(ind4),:)),[1 1 1],'k',0,0.3);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
%         jbfill(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:)+ (rest_SE_channels(Channels_nos(RPind),:)),...
%            rest_avg_channels(Channels_nos(RPind),:)- (rest_SE_channels(Channels_nos(RPind),:)),'r','k',0,0.3);
       
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),rest_time, rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,preprocessed_move_epochs(Channels_nos(RPind),:),'b',erp_time,standardize_move_epochs(Channels_nos(RPind),:),'k',erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        text(-2,-5,[EEG.chanlocs(Channels_nos(ind4)).labels],'Color','k','FontWeight','normal','FontSize',paper_font_size-1); % ', ' num2str(Channels_nos(ind4))
        set(gca,'YDir','reverse');
        %if max(abs(move_avg_channels(Channels_nos(RPind),:))) <= 6
%          if max((move_avg_channels(Channels_nos(ind4),:)+(move_SE_channels(Channels_nos(ind4),:)))) >= 6 || ...
%                  min((move_avg_channels(Channels_nos(ind4),:)-(move_SE_channels(Channels_nos(ind4),:)))) <= -6
%             axis([move_erp_time(1) move_erp_time(end) -15 15]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(gca,'FontWeight','bold');  
%             set(gca,'YTick',[-10 0 10]);
%             set(gca,'YTickLabel',{'-10'; '0'; '10'});
%         else
            axis([-2.5 1 -6 3]);
            %axis([move_erp_time(1) 1 -15 15]);
            set(gca,'YTick',[-5 0 2]);
            set(gca,'YTickLabel',{'-5'; '0'; '+2'},'FontWeight','normal','FontSize',paper_font_size-1);
%        end
        line([0 0],[-10 10],'Color','k','LineWidth',0.5,'LineStyle','--');  
        line([-2.5 1.5],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
        plot_ind4 = plot_ind4 + 1;
        
        if ind4 == 6
            set(gca,'XColor',[1 1 1],'YColor',[1 1 1])
            set(gca,'YtickLabel',' ');
             hylab = ylabel('MRCP Grand Average','FontSize',paper_font_size-1,'Color',[0 0 0]);
             pos_hylab = get(hylab,'Position');
             set(hylab,'Position',[pos_hylab(1) pos_hylab(2) pos_hylab(3)]);
        else
            set(gca,'Visible','off');
        end
        
    %    grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
     %   set(gca,'XTick',[-2 -1 0 1]);
     %   set(gca,'XTickLabel',{'-2';'-1'; '0';'1'});  
        
          
    end

    % subplot(4,3,5);
    % topoplot([],EEG.chanlocs,'style','blank','electrodes','labels','chaninfo',EEG.chaninfo);
    %subplot(5,5,8);
   
    
    axes(T_plot(11));
    set(gca,'Visible','on');
    bgcolor = get(gcf,'Color');
    set(gca,'YColor',[1 1 1]);
    set(gca,'XTick',[-2 0 1]);
    set(gca,'XTickLabel',{'-2'; 'MO';'1'},'FontSize',paper_font_size-1);  
    set(gca,'TickLength',[0.03 0.025])
    hold on;
    xlabel('Time (s)', 'FontSize',paper_font_size-1);
    
    % Annotate line
    axes(T_plot(15));
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
    
    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
         %tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.tif'];
         %fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.fig'];
        %print('-dtiff', '-r300', tiff_filename); 
        %saveas(gcf,fig_filename);
    else
        disp('Save figure aborted');
    end
    
    %% ERP image plots
%     mt1 = 101;  
%     peak_times = move_erp_time(mt1 + min_avg(:,2));
%     [sorted_peak_times,sorting_order] = sort(peak_times,'descend');
%     modified_EEG_data = EEG.data(:,:,sorting_order);
%     EEG.data = modified_EEG_data;
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
%     eeglab redraw;
    
    %Channels_nos = [9 32 10 44 48 14 49 15 19 53 20 54];
    ra1 = [-8 8];
    figure;
    T_plot = tight_subplot(5,5,0.05,[0.1 0.1],[0.1 0.1]);
    for ind4 = 1:length(Channels_nos)
        axes(T_plot(ind4));
        if ind4 == 25
            pop_erpimage(EEG,1, Channels_nos(ind4),[[]],'',5,1,{},[],'','noxlabel','cbar','on','caxis',[ra1(1) ra1(2)],'cbar_title','\muV'); %EEG.chanlocs(Channels_sel(ch)).labels
            
        else
            pop_erpimage(EEG,1, Channels_nos(ind4),[[]],'',5,1,{},[],'','noxlabel','cbar','off','caxis',[ra1(1) ra1(2)],'cbar_title','\muV'); %EEG.chanlocs(Channels_sel(ch)).labels
        end
    end
    set(gca,'XTick',[-2 -1 0 1 2 3]);
    set(gca,'XTickLabel',{'-2'; '-1'; '0';'1';'2';'3'});  
    %export_fig MS_ses1_cond1_block80_ERP '-png' '-transparent';
    %print -dtiff -r450 LSGR_cond3_erp.tif
   mtit([Subject_name ', Backdrive Mode, Day 1 & 2'],'fontsize',14,'yoff',0.025);
    
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
%% Save data for training classifier
if manipulate_epochs == 1
   Average.move_epochs = move_epochs;
   Average.rest_epochs = rest_epochs;
   Average.Fs_eeg = Fs_eeg;
   Average.apply_baseline_correction = apply_baseline_correction;

   Average.move_avg_channels = move_avg_channels;
   Average.move_std_channels = move_std_channels;
   Average.move_SE_channels = move_SE_channels;
   Average.move_mean_baseline = move_mean_baseline;   
   Average.move_erp_time = move_erp_time;     

   Average.rest_avg_channels = rest_avg_channels;
   Average.rest_std_channels = rest_std_channels;
   Average.rest_SE_channels = rest_SE_channels;
   Average.rest_mean_baseline = rest_mean_baseline;   
   Average.rest_erp_time = rest_erp_time;
   Average.remove_corrupted_epochs = remove_corrupted_epochs;
   
   Average.RP_chans = input('Enter channel numbers where RP is seen. Example - [1 2 3 4]: ');

   file_identifier = [];
   if use_noncausal_filter == 1
      file_identifier = [ file_identifier '_noncausal'];
   else
      file_identifier = [ file_identifier '_causal'];
   end

   filename1 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_average' file_identifier '.mat'];
   save(filename1,'Average');   
end
%% LDA Classifier training & validation
if train_LDA_classifier == 1
% State features to be used
avg_sensitivity = [];
avg_specificity = [];
avg_accur       = [];
% num_channels    = [];
avg_TPR = [];
avg_FPR = [];

% move_window = [-0.7 -0.1];
% rest_window = [-0.7 -0.1];
% classchannels = [48,14,49,9]; % C1, Cz, C2
% 
% mlim1 = abs(-2-(move_window(1)))*10+1;
% mlim2 = abs(-2-(move_window(2)))*10+1;
% rlim1 = abs(-1-(rest_window(1)))*10+1;
% rlim2 = abs(-1-(rest_window(2)))*10+1;
% 
% num_diff = 0;               % calculate slope of EEG signals
% gain = 10^num_diff;
% if gain == 0
%     gain = 1;
% end
% num_feature_per_chan = mlim2 - mlim1 + 1 - num_diff;
% data_set = zeros(2*no_epochs,length(classchannels)*(length(mlim1:mlim2)-num_diff));
% data_set_labels = zeros(2*no_epochs,1);
% 
% for train_ind = 1:no_epochs
%     if num_diff == 0
%         for chan_ind = 1:length(classchannels)
%             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
%             ra2 = chan_ind*num_feature_per_chan;
%             data_set(train_ind,ra1:ra2) = move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind));
%         end
%     else
%         for chan_ind = 1:length(classchannels)
%             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
%             ra2 = chan_ind*num_feature_per_chan;
%             data_set(train_ind,ra1:ra2) = diff(move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind)),num_diff,2).*gain;
%         end
% 
%     end
%     data_set_labels(train_ind,1) = 1;
%     
%     if num_diff == 0               
%        for chan_ind = 1:length(classchannels)
%             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
%             ra2 = chan_ind*num_feature_per_chan;
%             data_set(train_ind+no_epochs,ra1:ra2) = rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind));
%        end        
%     else 
%         for chan_ind = 1:length(classchannels)
%             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
%             ra2 = chan_ind*num_feature_per_chan;
%             data_set(train_ind+no_epochs,ra1:ra2) = diff(rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind)),num_diff,2).*gain;
%         end   
%     end
%     data_set_labels(train_ind+no_epochs,1) = 2;
% end

% % % Calculate mean, variance, etc and prepare feature vector matrix
% %  nc = length(classchannels);
% %  win_len = int32(size(data_set,2)/nc);
% %  feature_space = [];
% %  for ch = 1:nc
% %      range1 = int32((ch-1)*win_len + 1);
% %      range2 = int32(ch*win_len);
% %      % feature_space = [feature_space, mean(data_set(:,range1:range2),2)];
% %      % feature_space = [feature_space, std(data_set(:,range1:range2),0,2)];       % std(X,flag = 0, dim = 2)
% %     
% %      % Slopes of regression
% %      t_interval =  move_window(1):0.1:move_window(2);
% %      [R1,M1,B1] = regression(repmat(t_interval,size(data_set,1),1),data_set(:,range1:range2));
% %      feature_space = [feature_space, M1]; 
% %  end
 

CVO = cvpartition(no_epochs,'kfold',5);
sensitivity = zeros(1,CVO.NumTestSets);
specificity = zeros(1,CVO.NumTestSets);
accur = zeros(1,CVO.NumTestSets);
TPR = zeros(1,CVO.NumTestSets);
FPR = zeros(1,CVO.NumTestSets);

for cv_index = 1:CVO.NumTestSets
    trIdx = CVO.training(cv_index);
    teIdx = CVO.test(cv_index);
     [test_labels, Error, Posterior, LogP, OutputCoefficients] = ...
        classify(data_set([teIdx;teIdx],:),data_set([trIdx;trIdx],:),data_set_labels([trIdx;trIdx],1), 'linear');

%    [test_labels, Error, Posterior, LogP, OutputCoefficients] = ...
%       classify(feature_space(teIdx,:),feature_space(trIdx,:),data_set_labels(trIdx,1), 'linear');

    CM = confusionmat(data_set_labels([teIdx;teIdx],:),test_labels);
    sensitivity(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2)); % or True Positive Rate
    specificity(cv_index) = CM(2,2)/(CM(2,2)+CM(2,1)); % or True Negative Rate
    TPR(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
    FPR(cv_index) = CM(2,1)/(CM(2,2)+CM(2,1));
    accur(cv_index) = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2))*100;
    
end

avg_sensitivity = [avg_sensitivity; mean(sensitivity)];
avg_specificity = [avg_specificity; mean(specificity)];
avg_accur       = [avg_accur; mean(accur)];
avg_TPR         = [avg_TPR; mean(TPR)];
avg_FPR         = [avg_FPR; mean(FPR)];
% num_channels    = [num_channels; length(classchannels)];
% num_channels    = [num_channels; num_feature_per_chan];
%% Find Classifier coefficients with 'best' performance. 
% performance = abs(sensitivity - specificity);
% best_index = find(performance == min(performance));
% 
% % Recalculate coefficients for best performance classifier
% trIdx = CVO.training(best_index);
% teIdx = CVO.test(best_index);
% 
% [test_labels, Error, Posterior, LogP, Best_OutputCoefficients] = ...
%        classify(data_set([teIdx;teIdx],:),data_set([trIdx;trIdx],:),data_set_labels([trIdx;trIdx],1), 'linear');
% 
% CM = confusionmat(data_set_labels([teIdx;teIdx],:),test_labels);
% best_sensitivity = CM(1,1)/(CM(1,1)+CM(1,2));
% best_specificity = CM(2,2)/(CM(2,2)+CM(2,1));
% 
% % Write Coefficients to a text file
% K12 = Best_OutputCoefficients(1,2).const;
% L12 = Best_OutputCoefficients(1,2).linear;
% 
% % Find margins on classifier's decision
% for len = 1:size(data_set,1)
%     classifier_decision(len) = K12+data_set(len,:)*L12;
% end
% 
% coeff_filename = [folder_path Subject_name '_coeffs.txt'];
% fileid = fopen(coeff_filename,'w+');
% fprintf(fileid,'**** Calibration Data for Subject %s *****\r\n',Subject_name);
% fprintf(fileid,'Avg_Sensitivity = %.2f \r\n',avg_sensitivity*100);
% fprintf(fileid,'Avg_Specificity = %.2f \r\n',avg_specificity*100);
% fprintf(fileid,'Best_Sensitivity = %.2f \r\n',best_sensitivity*100);
% fprintf(fileid,'Best_Specificity = %.2f \r\n',best_specificity*100);
% fclose(fileid);
% dlmwrite(coeff_filename,[K12;L12],'-append','delimiter','\t','precision','%.4f');
% 
% % Write Class Channels to a separate text file
% channels_filename = [folder_path Subject_name '_channels.txt'];
% dlmwrite(channels_filename,classchannels,'delimiter','\t','precision','%d');

end
%% SVM Classifier training & validation
%% Close-loop BCI testing in realtime.
if test_classifier == 1
   
   % if ~isempty(instrfind)
        % set and open serial port
        %obj = serial('com1','baudrate',115200,'parity','none','databits',8,'stopbits',1);   %InMotion
        obj = serial('com20','baudrate',19200,'parity','none','databits',8,'stopbits',1); %Mahi               % change9
        fopen(obj);
    %else
       % obj = -1;
        %disp('Serial Device Not FOUND!!');
    %end
% obj = 1;
   
    if use_GUI_for_testing == 1
        %--------------- Call GUI
        user.calibration_data.subject_initials = Subject_name;
        user.calibration_data.sess_num = Sess_num;
        user.calibration_data.block_num = Block_num;
        user.calibration_data.cond_num = Cond_num;
        user.calibration_data.folder_path = folder_path;
        user.testing_data.closeloop_sess_num = closeloop_Sess_num;
        user.testing_data.closeloop_block_num = 0;
        user.testing_data.closeloop_folder_path = closeloop_folder_path;
      
    [Proc_EEG,Ovr_Spatial_Avg,All_Feature_Vec,GO_Prob,Num_Move_Counts,Markers,all_cloop_prob_threshold, all_cloop_cnts_threshold,Proc_EMG] = BMI_Mahi_Closeloop_GUI(user,obj);
    
    else        
        % use script
        filename2 = [folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_causal.mat'];
        load(filename2);    
        [Proc_EEG,Ovr_Spatial_Avg,All_Feature_Vec,GO_Prob,Num_Move_Counts,Markers,cloop_prob_threshold, cloop_cnts_threshold] = BMI_Mahi_Closeloop(Performance,obj);
        all_cloop_prob_threshold = cloop_prob_threshold.*ones(1,length(All_Feature_Vec));
        all_cloop_cnts_threshold = cloop_cnts_threshold.*ones(1,length(All_Feature_Vec));

        closeloop_Block_num = input('Enter Trial/Block Number:');
        % Save variables
        var_filename = [closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_block' num2str(closeloop_Block_num) '_closeloop_results.mat'];
        save(var_filename,'Proc_EEG','Proc_EMG','Ovr_Spatial_Avg','All_Feature_Vec','GO_Prob','Num_Move_Counts','Markers','cloop_prob_threshold','cloop_cnts_threshold');
    end
       
 %   if exist('obj','var')
  %      if obj ~= -1
            % clear serial port
            fclose(obj);
            delete(obj);
            clear('obj');
   %     end
   % end
    % Useful serial port command
    %instrfind, delete(instrfindall)
    
     
    
%% Raster plot for close loop

use_GUI_for_testing = 0;

if use_GUI_for_testing == 0
    raw_Fs = 500;
    proc_Fs = 20;
    downsamp_factor = raw_Fs/proc_Fs;
    
    if ~exist('Markers')
        Proc_EEG = processed_eeg; 
        %Proc_EMG = downsample(processed_emg',downsamp_factor)';
        Proc_EMG = resample(processed_emg',6,1);
        Ovr_Spatial_Avg = Overall_spatial_chan_avg;
        All_Feature_Vec = all_feature_vectors;
        GO_Prob = GO_Probabilities;
        Num_Move_Counts = move_counts;
        Markers = marker_block;

    end

    event_times = (double(Markers(:,1))/raw_Fs);    
    pred_start_stop_times = round(event_times(Markers(:,2)==50).*proc_Fs)/proc_Fs;
    if length(pred_start_stop_times) == 1
        pred_start_stop_times = [pred_start_stop_times(1) round(event_times(end).*proc_Fs)/proc_Fs];
    end

     pred_start_stop_times = [pred_start_stop_times(1)  pred_start_stop_times(2)-1];
     
    eeg_time = (0:1/proc_Fs:length(Ovr_Spatial_Avg)/proc_Fs) + pred_start_stop_times(1);
    eeg_time = round(eeg_time.*100)/100;

    pred_stimulus_times = round(event_times(Markers(:,2)==200).*proc_Fs)/proc_Fs;
    pred_GO_times = round(event_times(Markers(:,2)==300).*proc_Fs)/proc_Fs;
    pred_EMG_GO_times = round(event_times(Markers(:,2)==400).*proc_Fs)/proc_Fs;
    pred_move_onset_times = round(event_times(Markers(:,2)==100).*proc_Fs)/proc_Fs;

    pred_Ovr_Spatial_Avg = Ovr_Spatial_Avg(find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2))); % Correct the '-7'
    pred_eeg_time = eeg_time(find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2)));
    pred_Proc_EEG = Proc_EEG(:,find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2)));
    pred_Proc_EMG = Proc_EMG(:,find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2)));
    
    if length(All_Feature_Vec) > length(pred_Ovr_Spatial_Avg)
        All_Feature_Vec = All_Feature_Vec(:,1:length(pred_Ovr_Spatial_Avg));
        GO_Prob = GO_Prob(1:length(pred_Ovr_Spatial_Avg));
        Num_Move_Counts = Num_Move_Counts(1:length(pred_Ovr_Spatial_Avg));
    else
        pred_Ovr_Spatial_Avg = pred_Ovr_Spatial_Avg(1:length(All_Feature_Vec));
        pred_eeg_time = pred_eeg_time(1:length(All_Feature_Vec));
        pred_Proc_EEG = pred_Proc_EEG(:,1:length(All_Feature_Vec));
        pred_Proc_EMG = pred_Proc_EMG(:,1:length(All_Feature_Vec));
    end

    raster_data = [All_Feature_Vec(1:3,1:length(pred_Ovr_Spatial_Avg)); [zeros(1,50) All_Feature_Vec(4,51:length(pred_Ovr_Spatial_Avg))]; pred_Ovr_Spatial_Avg;...
                                            [zeros(2,50) pred_Proc_EMG(:,51:length(pred_Ovr_Spatial_Avg))]]'; % Needs to be corrected
    raster_zscore = zscore(raster_data);
    raster_time = pred_eeg_time;
    raster_colors = ['m','k','b','r','k','b','r'];
    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 5;
    %raster_ylim1 = 0;
    %raster_ylim2 = (raster_col+1)*add_offset;
    figure;
    raster_zscore(:,1:3) = raster_zscore(:,1:3).*0.5;
    raster_zscore(:,5) = raster_zscore(:,5).*0.5;
    raster_zscore(:,4) = raster_zscore(:,4).*0.5;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index+20;  % Add offset to each channel of raster plot
        myhand(raster_index) = plot(raster_time,raster_zscore(:,raster_index),raster_colors(raster_index));
        hold on;
    end
    set(myhand(5),'LineWidth',2);
    set(myhand(6),'LineWidth',2);
    set(myhand(7),'LineWidth',2);
    %plot(raster_time, 10/3*Num_Move_Counts, '--b','LineWidth',2);
    hold on;
    plot(raster_time, 10 + 10*GO_Prob(1:length(pred_Ovr_Spatial_Avg)), '.r','LineWidth',2);
    plot(raster_time, 10 + 10*all_cloop_prob_threshold(1:length(pred_Ovr_Spatial_Avg)),'-','Color',[0.4 0.4 0.4],'LineWidth',2);
    plot(raster_time, all_cloop_cnts_threshold(1:length(pred_Ovr_Spatial_Avg)),'-k','LineWidth',2);
    plot(raster_time, Num_Move_Counts,'b');
    %line([0 200],10.*[cloop_prob_threshold cloop_prob_threshold],'Color',[0.4 0.4 0.4],'LineWidth',2);
    axis([pred_start_stop_times(1) pred_start_stop_times(2) 1 60]);
    %ylim([0 40]);

    myaxis = axis;
%     for plot_ind1 = 1:length(pred_stimulus_times);
%         line([pred_stimulus_times(plot_ind1), pred_stimulus_times(plot_ind1)],[myaxis(3)-1, myaxis(4)],'Color','b','LineWidth',1);
%         text(pred_stimulus_times(plot_ind1)-1,myaxis(4)+0.5,'Start','Rotation',60,'FontSize',12);
%         %text(pred_stimulus_times(plot_ind1)-1.5,myaxis(4)+1,'shown','Rotation',0,'FontSize',12);
%         hold on;
%     end

    for plot_ind2 = 1:length(pred_GO_times);
        line([pred_GO_times(plot_ind2), pred_GO_times(plot_ind2)],[myaxis(3)-1, myaxis(4)],'Color','k','LineWidth',1,'LineStyle','--');
        text(pred_GO_times(plot_ind2),myaxis(4)+0.5,'EEG','Rotation',60,'FontSize',12);
        hold on;
    end

    for plot_ind4 = 1:length(pred_EMG_GO_times);
                        line([pred_EMG_GO_times(plot_ind4), pred_EMG_GO_times(plot_ind4)],[myaxis(3)-1, myaxis(4)],'Color','b','LineWidth',2,'LineStyle','--');
                        text(pred_EMG_GO_times(plot_ind4),myaxis(4)+0.5,'EEG+EMG','Rotation',60,'FontSize',12);
                        hold on;
    end
                    
%     for plot_ind3 = 1:length(pred_move_onset_times);
%         line([pred_move_onset_times(plot_ind3), pred_move_onset_times(plot_ind3)],[myaxis(3), myaxis(4)],'Color','r','LineWidth',2,'LineStyle','--');
%         hold on;
%     end
    set(gca,'YTick',[1 all_cloop_cnts_threshold(1) 5 10 10+10*all_cloop_prob_threshold(1) 20 25 30 35 40 45 50 55]);
    set(gca,'YTickLabel',{'1' 'Count_Thr' '5' 'p(GO) = 0','Prob_Thr','p(GO) = 1', 'AUC', '-ve Peak','Slope','Mahalanobis','Spatial Avg EEG','Biceps','Triceps'});
    xlabel('Time (sec.)', 'FontSize' ,12);
    %export_fig 'Block8_results' '-png' '-transparent'
    %%
    % Plot EMG data for deciding EMG Thresholds
    figure; 
    plot(eeg_time,Proc_EMG(1,end-length(eeg_time)+1:end),'k','LineWidth',1.5); hold on;
    plot(eeg_time,Proc_EMG(2,end-length(eeg_time)+1:end),'r','LineWidth',1.5); hold on;
    axis([eeg_time(1) eeg_time(end) 0 40]);
    myaxis = axis;
    
    for plot_ind2 = 1:length(pred_GO_times);
        line([pred_GO_times(plot_ind2), pred_GO_times(plot_ind2)],[myaxis(3)-1, myaxis(4)],'Color','k','LineWidth',1,'LineStyle','--');
        text(pred_GO_times(plot_ind2),myaxis(4)+0.5,'EEG','Rotation',60,'FontSize',12);
        hold on;
    end

    for plot_ind4 = 1:length(pred_EMG_GO_times);
                        line([pred_EMG_GO_times(plot_ind4), pred_EMG_GO_times(plot_ind4)],[myaxis(3)-1, myaxis(4)],'Color','b','LineWidth',2,'LineStyle','--');
                        text(pred_EMG_GO_times(plot_ind4),myaxis(4)+0.5,'EEG+EMG','Rotation',60,'FontSize',12);
                        hold on;
    end
                    
%     for plot_ind3 = 1:length(pred_move_onset_times);
%         line([pred_move_onset_times(plot_ind3), pred_move_onset_times(plot_ind3)],[myaxis(3), myaxis(4)],'Color','r','LineWidth',2,'LineStyle','--');
%         hold on;
%     end
    xlabel('Time (sec.)', 'FontSize' ,12);
    
end
end
disp('*************** END OF PROGRAM, Have A Good Day !!***********************************');
%% OLD - Extract epochs between stimulus and response events

% % trials(1).eeg = [];
% % trials(1).target = [];
% % trials(1).position = [];
% trials = struct('eeg',{},'target',{});
% movements = struct('eeg',{},'target',{});
% 
% target_index = 1;
% for trial_cnt = 1:(length(epoch_indices)-1)     % Extract only 100 trials !!
%     trials(trial_cnt).eeg = detrend(EEG.data(:, ...
%     round(stimulus_trig(epoch_indices(trial_cnt,1))*Fs_eeg): ...
%     round(response_trig(epoch_indices(trial_cnt,2))*Fs_eeg)));
% 
%     movements(trial_cnt).eeg = detrend(EEG.data(:, ...
%     round(response_trig(epoch_indices(trial_cnt,2))*Fs_eeg): ...
%     round(stimulus_trig(epoch_indices(trial_cnt+1,1))*Fs_eeg)));
% 
%     if mod(trial_cnt,2) == 0
%         trials(trial_cnt).target = 0;
%         movements(trial_cnt).target = 0;
%     else
%         trials(trial_cnt).target = targets(target_index);
%         movements(trial_cnt).target = targets(target_index);
%         target_index = target_index + 1;
%     end    
% end
% 
% 
% 
% 
% %         
% % for y = 1:2*length(targets) 
% %     myvar(y,1) = trials(y).target;
% % end
% 
        
% %% OLD - Separate trials based on target
% C_trials = struct('eeg',{},'target',{});
% N_trials = struct('eeg',{},'target',{});
% S_trials = struct('eeg',{},'target',{});
% E_trials = struct('eeg',{},'target',{});
% W_trials = struct('eeg',{},'target',{});
% C_movements = struct('eeg',{},'target',{});
% N_movements = struct('eeg',{},'target',{});
% S_movements = struct('eeg',{},'target',{});
% E_movements = struct('eeg',{},'target',{});
% W_movements = struct('eeg',{},'target',{});
% 
% 
% 
% for count = 1:(length(epoch_indices)-1)
%     switch trials(count).target
%         case 0
%             C_trials = [C_trials trials(1,count)];
%             C_movements = [C_movements movements(1,count)];
%         case 1
%             S_trials = [S_trials trials(1,count)];
%             S_movements = [S_movements movements(1,count)];
%             
%         case 2
%             W_trials = [W_trials trials(1,count)];
%             W_movements = [W_movements movements(1,count)];
%             
%         case 3
%             N_trials = [N_trials trials(1,count)];
%             N_movements = [N_movements movements(1,count)];
%             
%         case 4
%             E_trials = [E_trials trials(1,count)];
%             E_movements = [E_movements movements(1,count)];
%             
%         otherwise warning('Error in switch');
%     end
% end
% 
%% ICA kurtosis
% for i = 64:-1:21
%     pop_signalstat( EEG, 0, i );
% end
