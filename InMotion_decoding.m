% Processing data from InMotion and EEG
% By Nikunj Bhagat, 8/14/13; 12/24/13; 1/23/14;
% 2/8/14 - Replacing EEGLAB functions.
%        - No standardization, extract targets option. 
clear;
%close all;
%% Global variables 
myColors = ['g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m',...
    'g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];
Subject_name = 'FJ2';
Sess_num = 2;
Block_num = [];
folder_path = ['F:\Nikunj_Data\InMotion_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
% folder_path = ['D:\Subject_' Subject_name '\LG_Session4\'];
Channels_nos = [4, 38, 5, 39, 6, 43, 9, 32, 10, 44, 13, 48, 14, 49, 15, 52, 19, ... 
                53, 20, 54, 24, 57, 25, 58, 26];    % 32 or 65 for FCz
            
% Flags to control the processing 
train_LDA_model = 1;
test_LDA_model = 0;

if train_LDA_model == 1
    disp('******************** Training Model **********************************');
    %standardize_data_flag = 0;
    process_raw_eeg = 0;
    label_events = 0;       % process kinematics and label the events/triggers
    include_target_events = 0;
    extract_epochs = 1;     % extract move and rest epochs
    plot_ERPs = 1;
    train_classifier = 0;
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB under Matlab 
    
elseif test_LDA_model == 1
    disp('******************** Close-loop Model testing *************************');
    %standardize_data_flag = 0;
    process_raw_eeg = 0;
    label_events = 0;       % process kinematics and label the events/triggers
    %include_target_events = 0;
    extract_epochs = 0;     % extract move and rest epochs
    plot_ERPs = 0;    
end

% Miscellaneous Flags
    common_avg_ref = 1;
    car_chnns_eliminate = [1 2 41 46];
    %car_chnns_eliminate = [];
    
    large_laplacian_ref  = 0;
    small_laplacian_ref = 0;
    weighted_avg_ref = 1;
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

hpfc = 0.1;     % Cutoff frequency = 0.1 Hz    
lpfc = 1;       % Cutoff frequency = 1 Hz

%% Preprocessing of raw EEG signals
% % Inputs - SB_raw.set
% % Outputs - SB_preprocessed.set; SB_standardized.set

if process_raw_eeg == 1
    
    % ****STEP 0: Load EEGLAB dataset (SB_raw.set)
    % read in the dataset
    if ~isempty(Block_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_raw.set'], folder_path); 
    else
        EEG = pop_loadset( [Subject_name '_raw.set'], folder_path); 
    end
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    %eeglab redraw;
    
    % Define Paramteres
    raw_eeg_Fs = EEG.srate;
    raw_eeg = EEG.data;
    [eeg_nbchns,eeg_pnts] = size(EEG.data);
    
    % ****STEP 1: Plot/Scroll Data in EEGLAB and manually remove noisy
    %         channels/segments of data.


    % Additional STEP: Perform ICA analysis and remove artifacts

    % ****STEP 2: High Pass Filter data, 0.1 Hz, 4th order Butterworth
    
    [num_hpf,den_hpf] = butter(4,(hpfc/(raw_eeg_Fs/2)),'high');
    eeg_hpf = dfilt.df2(num_hpf,den_hpf);       % Create filter object. Not essential  
    %   figure;
    %   freqz(num_hpf,den_hpf,512,raw_eeg_Fs); % Better to use fvtool
    %fvtool(eeg_hpf,'Analysis','freq','Fs',raw_eeg_Fs);  % Essential to visualize frequency response properly.
    HPFred_eeg = zeros(size(raw_eeg));
    for i = 1:eeg_nbchns
        %EEG.data(i,:) = detrend(EEG.data(i,:)); %Commented on 12-9-13
        %HPFred_eeg(i,:) = filtfilt(num_hpf,den_hpf,double(EEG.data(i,:)));
        HPFred_eeg(i,:) = filtfilt(eeg_hpf.sos.sosMatrix,eeg_hpf.sos.ScaleValues,double(raw_eeg(i,:)));
    end
    %EEG.data = HPFred_eeg;

    % ****STEP 3: Re-Reference data
    SPFred_eeg = HPFred_eeg;
    
    if common_avg_ref == 1
%         %EEG = pop_reref( EEG, [],'exclude',[1 2 17]);       % Exclude HEOG, VEOG channels
%         EEG = pop_reref( EEG, []);
%         % EEG.setname='NB_CAR';
%         EEG = eeg_checkset( EEG );
        
        % Create Common Avg Ref matrix of size 64x64
        total_chnns = eeg_nbchns;
        num_car_channels = total_chnns - length(car_chnns_eliminate);          % Number of channels to use for CAR
        M_CAR =  (1/num_car_channels)*(diag((num_car_channels-1)*ones(total_chnns,1)) - (ones(total_chnns,total_chnns)-diag(ones(total_chnns,1))));
        
        if ~isempty(car_chnns_eliminate)
            for elim = 1:length(car_chnns_eliminate)
                M_CAR(:,car_chnns_eliminate(elim)) = 0;
            end
        end
        
        SPFred_eeg = M_CAR*(SPFred_eeg);
    end 
    if large_laplacian_ref == 1
%         EEG = exp_eval(flt_laplace(EEG,4));
%         EEG.ref = 'laplacian';
%         EEG.setname = 'laplacian-ref';
%         EEG = eeg_checkset( EEG );      
    SPFred_eeg = spatial_filter(SPFred_eeg,'LLAP');          
    end 
    
    if small_laplacian_ref == 1
    SPFred_eeg = spatial_filter(SPFred_eeg,'SLAP');    
    end
    
    if weighted_avg_ref == 1
    SPFred_eeg = spatial_filter(SPFred_eeg,'WAVG');
    end
        
    % ****STEP 4: Low Pass Filter data, 1 Hz, 8th order Butterworth
    
    [num_lpf,den_lpf] = butter(4,(lpfc/(raw_eeg_Fs/2)));
    eeg_lpf = dfilt.df2(num_lpf,den_lpf);       % Create filter object. Not essential
    %   figure;
    %   freqz(num_lpf,den_lpf,512,raw_eeg_Fs);  % Better to use fvtool
    %   fvtool: http://www.mathworks.com/help/signal/ref/fvtool.html#f7-1176930
    % fvtool(eeg_lpf,'Analysis','freq','Fs',raw_eeg_Fs);  % Essential to visualize frequency response properly.
    LPFred_eeg = zeros(size(SPFred_eeg));
    for i = 1:eeg_nbchns
        %LPFred_eeg(i,:) = filtfilt(num_lpf,den_lpf,double(EEG.data(i,:)));
        LPFred_eeg(i,:) = filtfilt(eeg_lpf.sos.sosMatrix,eeg_lpf.sos.ScaleValues,double(SPFred_eeg(i,:)));
    end
    EEG.data = LPFred_eeg;   
    
    % **** STEP 5: Resample to 100 Hz 
    %Preproc_eeg = decimate(LPFred_eeg,5);
    
    EEG = pop_resample( EEG, 100);
    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_preprocessed'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    if ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_preprocessed.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_preprocessed.set'],'filepath',folder_path);
    end
    
    EEG = eeg_checkset( EEG );
    Fs_eeg = EEG.srate;           % Sampling rate after downsampling

    %Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    % **** STEP 6: Compute Power Spectral Density of EEG Channels

    NFFT = 512; % Freq_Res = Fs/NFFT
    PSD_eeg = zeros(length(Channels_nos),((NFFT/2)+1));
    PSD_f = zeros(1,((NFFT/2)+1));
    figure; hold on; grid on;
    for psd_len = 1:length(Channels_nos)
        [PSD_eeg(psd_len,:),PSD_f] = pwelch(detrend(double(EEG.data(Channels_nos(psd_len),:))),[],[],NFFT,Fs_eeg);   
        plot(PSD_f(:),10*log10(PSD_eeg(psd_len,:)),myColors(psd_len));
        hold on;
    end
    xlabel('Frequency Hz');
    ylabel('PSD dB');
    title('Preprocessed EEG Channels PSD');
    hold off
    % % Create Window function with unity area
    % % NOTE : Windowing is useful when plotting PSD of raw EEG signals
    % % N = input('Enter Window Length: ') % Larger N means MORE smoothing
    % N = 5;
    % w = ones(1,N)/N;
    % 
    % % Convolve with the window function
    % PSD_filtered = conv(PSD_eeg(1,:),w);
    % 
    % % Remove Excess Data points
    % PSD_filtered = PSD_filtered(ceil(N/2):end-floor(N/2));

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

%% Processing of InMotion Kinematics
% ********** Matrix Columns ****************
% [time PosX PosY VelX VelY Shoulder_Angle Elbow_Angle dout0 dout1 Fx Fy Fz ??]
% *************
if label_events == 1
    
    %kinematics_raw = importdata([Subject_name '_kinematics.mat']); % Raw data sampled at 200 Hz
    kinematics_raw = importdata([folder_path Subject_name '_ses1_training_kinematics.txt']); % Raw data sampled at 200 Hz
    Fs_kin_raw = 200;
    % Set zero time
    t0_kin = kinematics_raw(1,1);
    kinematics_raw(:,1) = kinematics_raw(:,1)-t0_kin;
    kinematics_raw(:,1) = kinematics_raw(:,1)./Fs_kin_raw;

    % Downsample to 100 Hz; better than resample() 
    kinematics = downsample(kinematics_raw,2); % Decimation by 1/2
    Fs_kin = 100;

    % Low Pass Filter Position and Velocity
    [fnum,fden] = butter(4,4/(Fs_kin/2),'low');
    %freqz(fnum,fden,128,Fs_kin);
    position_f(:,1) = filtfilt(fnum,fden,kinematics(:,2));
    position_f(:,2) = filtfilt(fnum,fden,kinematics(:,3));
    velocity_f(:,1) = filtfilt(fnum,fden,kinematics(:,4));
    velocity_f(:,2) = filtfilt(fnum,fden,kinematics(:,5));
    tan_velocity = sqrt(velocity_f(:,1).^2 + velocity_f(:,2).^2);

    %****** Extract unique triggers and modify External Triggers to include only first sample instant
    % figure; plot(kinematics(:,1),kinematics(:,8));
    temp_response_trig = kinematics(:,9);
    temp_stimulus_trig = kinematics(:,8);
    % % Commeneted for new data - 9/5/13
    % [kin_response_trig_index,kinematics(:,8)] = ExtractUniqueTriggers(kinematics(:,8),1);
    [kin_response_trig_index,kinematics(:,8)] = ExtractUniqueTriggers(temp_response_trig,1); 

    %hold on; plot(kinematics(:,1),kinematics(:,8),'r');
    %figure; plot(kinematics(:,1),kinematics(:,9));

    % % Commeneted for new data - 9/5/13
    %[kin_stimulus_trig_index,kinematics(:,9)] = ExtractUniqueTriggers(kinematics(:,9),1); 
    [kin_stimulus_trig_index,kinematics(:,9)] = ExtractUniqueTriggers(temp_stimulus_trig,1); 
    %hold on; plot(kinematics(:,1),kinematics(:,9),'r');

    % stamp_dout0 = kin_response_trig_index/200;
    % f_stamp_dout0 = [0; stamp_dout0(1:(length(stamp_dout0)-1),1)];
    % dout0_diff = stamp_dout0 - f_stamp_dout0;
    % 
    % stamp_dout1 = kin_stimulus_trig_index/200;
    % f_stamp_dout1 = [0; stamp_dout1(1:(length(stamp_dout1)-1),1)];
    % dout1_diff = stamp_dout1 - f_stamp_dout1;

    % % kin_epoch_indices = [];
    % % kin_s = 2;
    % % for kin_r = 2:length(kin_response_trig_index)
    % %     if ((kin_stimulus_trig_index(kin_s) < kin_response_trig_index(kin_r)) && (kin_stimulus_trig_index(kin_s) > (kin_response_trig_index(kin_r)-(5*Fs_kin)))) %Does stimulus_trig occurs within 5 sec of response_trig
    % %         % epoch_times(k_s,:) = [stimulus_trig(k_s) response_trig(k_r)];
    % %         kin_epoch_indices = [kin_epoch_indices; kin_s kin_r];
    % %         kin_s = kin_s+1;
    % %     end
    % % end



%% Extract trigger latencies from EEG signals
% Ensure SB_preprocessed.set/SB_standardized.set is loaded into EEGLAB

if standardize_data_flag == 0
    EEG = pop_loadset( [Subject_name '_preprocessed.set'], folder_path); % read in the dataset
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;
else
    EEG = pop_loadset( [Subject_name '_standardized.set'], folder_path); % read in the dataset
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;
end

Fs_eeg = EEG.srate;
eeg_response_trig = [];     % Trigger event when robot moves
eeg_stimulus_trig = [];     % Trigger event when target appears


for j=1:length(EEG.event)
    if (strcmp(EEG.event(j).type,'R128'))
        eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
    elseif (strcmp(EEG.event(j).type,'S  2'))
        eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
    end
end
eeg_response_trig = eeg_response_trig/Fs_eeg;       % Convert to seconds.
eeg_stimulus_trig = eeg_stimulus_trig/Fs_eeg;
stimulus_time_correction = eeg_stimulus_trig(1,1);  % Save value for later
response_time_correction = eeg_response_trig(1,1);  % Save value for later
eeg_stimulus_trig = eeg_stimulus_trig - stimulus_time_correction;   % Apply correction, make t = 0
eeg_response_trig = eeg_response_trig - response_time_correction;   % Apply correction, make t = 0  

% Round triggers upto two decimal places
eeg_stimulus_trig = (ceil(eeg_stimulus_trig.*100))/100;
eeg_response_trig = (ceil(eeg_response_trig.*100))/100;

% % Commented for new data - 9/5/13
% valid_cnt = 0;          % Number of valid trials 
% tan_velocity_epoch = [];
% Move_onset_indices = zeros(length(eeg_stimulus_trig),1);
% Move_onset_trig = [[] []];
% for trig_cnt = 2:length(eeg_stimulus_trig)
%     limit1 = find(kinematics(:,1)==eeg_stimulus_trig(trig_cnt-1)) + Fs_eeg;            % Wait 1 sec after end of previous movement
%     limit2 = find(kinematics(:,1)==eeg_stimulus_trig(trig_cnt));
%     tan_velocity_epoch = tan_velocity(limit1:limit2);
%     tan_velocity_max   = max(tan_velocity_epoch);
%     vel_threshold      = 0.05*tan_velocity_max;
%     Move_onset_indices(trig_cnt) = find(tan_velocity_epoch >= vel_threshold,1,'first') + limit1 -1; %Important to add limit1
%     if((Move_onset_indices(trig_cnt) - limit1) >= 2*Fs_eeg)     % Wait atleast 2.5 secs after end of previous movement.
%         valid_cnt = valid_cnt + 1;
%         Move_onset_trig = [Move_onset_trig; [100 kinematics(Move_onset_indices(trig_cnt),1)]];   % '100' is for appending event to EEG dataset
%     end
% end

valid_cnt = 0;          % Number of valid trials 
rest_cnt = 0;
tan_velocity_epoch = [];
Move_onset_indices = zeros(length(eeg_response_trig),1);
Move_onset_trig = [[] []];
Target_cue_trig = [[] []];
stimulus_ind = 2;
for trig_cnt = 2:length(eeg_response_trig)
    limit1 = find(kinematics(:,1)==eeg_stimulus_trig(stimulus_ind-1));            % Not required - Wait 1 sec after end of previous movement
    limit2 = find(kinematics(:,1)==eeg_response_trig(trig_cnt));
    if limit1 < limit2
        tan_velocity_epoch = tan_velocity(limit1:limit2);
        tan_velocity_max   = max(tan_velocity_epoch);
        vel_threshold      = 0.05*tan_velocity_max;
        Move_onset_indices(trig_cnt) = find(tan_velocity_epoch >= vel_threshold,1,'first') + limit1 -1; %Important to add limit1
        if((Move_onset_indices(trig_cnt) - limit1) >= 2.0*Fs_eeg)     % Wait atleast 2 secs after end of previous movement.
            valid_cnt = valid_cnt + 1;
            Move_onset_trig = [Move_onset_trig; [100 kinematics(Move_onset_indices(trig_cnt),1)]];   % '100' is for appending event to EEG dataset
            Target_cue_trig = [Target_cue_trig; [200 kinematics(limit1,1)]];   % '200' is for appending event to EEG dataset
        end
    else
        rest_cnt = rest_cnt + 1;
        stimulus_ind = stimulus_ind - 1;
    end
    stimulus_ind = stimulus_ind + 1;
end

% Plot Stimulus & Movement onset triggers on tangential velocity
lastpt = length(kinematics(:,1));
figure; hold on;
plot(kinematics(1:lastpt,1),tan_velocity(1:lastpt),'k');
myaxis = axis;
for plot_ind1 = 1:length(eeg_stimulus_trig);
    line([eeg_stimulus_trig(plot_ind1), eeg_stimulus_trig(plot_ind1)],[myaxis(3), myaxis(4)],'Color','g');
    hold on;
end
for plot_ind6 = 1:length(eeg_response_trig);
    line([eeg_response_trig(plot_ind6), eeg_response_trig(plot_ind6)],[myaxis(3), myaxis(4)],'Color','b');
    hold on;
end
for plot_ind2 = 1:length(Move_onset_trig);
    line([Move_onset_trig(plot_ind2,2), Move_onset_trig(plot_ind2,2)],[myaxis(3), myaxis(4)],'Color','r');
    hold on;
end
% 
% for plot_ind7 = 1:length(Target_cue_trig);
%     line([Target_cue_trig(plot_ind7,2), Target_cue_trig(plot_ind7,2)],[myaxis(3), myaxis(4)],'Color','g');
%     hold on;
% end
% hold off;


%% Generate Raster Plot

    
    raster_colors = ['g','r','b','k','y','c','m','g','r','b','k','r','b','k','g','r','b','k','y','c','m'];
    raster_Fs = Fs_kin;
    first_trig = 10;
    last_trig  = 15;
    raster_lim1 = kin_stimulus_trig_index(first_trig)-Fs_kin;   % Take previous 1 second
    raster_lim2 = kin_stimulus_trig_index(last_trig)+Fs_kin;   % Take next 1 second
    raster_time = kinematics(raster_lim1:raster_lim2,1);
    eeg_raster_lim1 = round(stimulus_time_correction*100)+raster_lim1;
    eeg_raster_lim2 = eeg_raster_lim1 + (raster_lim2-raster_lim1);  %Matrix dimension mismatch

    % Plot selected EEG channels and kinematics
    raster_data = zeros(length(position_f(raster_lim1:raster_lim2,1)),length(Channels_nos));
    for chan_index = 1:length(Channels_nos)
        raster_data(:,chan_index) = EEG.data(Channels_nos(chan_index),eeg_raster_lim1:eeg_raster_lim2)';
    end
    raster_data = [ raster_data position_f(raster_lim1:raster_lim2,1) position_f(raster_lim1:raster_lim2,2) ...
        tan_velocity(raster_lim1:raster_lim2) ];

    % Standardization/Normalization of Signals; 
    % http://en.wikipedia.org/wiki/Standard_score
    % Standard score or Z-score is the number of standard deviations an
    % obervation is above/below the mean. 
    % z-score = (X - mu)/sigma;

    raster_zscore = zscore(raster_data);

    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 5;
    raster_ylim1 = 0;
    raster_ylim2 = (raster_col+1)*add_offset;
    figure;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
        plot(raster_time,raster_zscore(:,raster_index),raster_colors(raster_index));
        hold on;
    end
    for plot_ind1 = first_trig:last_trig;          % Plot & label the stimulus triggers
        line([kin_stimulus_trig_index(plot_ind1)/raster_Fs, kin_stimulus_trig_index(plot_ind1)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','r','LineWidth',1.5,'LineStyle','-');
        text(kin_stimulus_trig_index(plot_ind1)/raster_Fs,raster_ylim2,'New Target','Rotation',60,'FontSize',9);
        hold on;
    end
    % % for plot_ind2 = first_trig:(last_trig-1);          % Plot & label the response triggers
    % %     line([kin_response_trig_index(plot_ind2)/raster_Fs, kin_response_trig_index(plot_ind2)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','r','LineWidth',1);
    % %     text(kin_response_trig_index(plot_ind2)/raster_Fs,raster_ylim2,'Move','Rotation',60,'FontSize',8);
    % %     hold on;
    % % end
    for plot_ind2 = first_trig+1:last_trig;          % Plot & label the response triggers
        line([kin_response_trig_index(plot_ind2)/raster_Fs, kin_response_trig_index(plot_ind2)/raster_Fs],[raster_ylim1,raster_ylim2],'Color','b','LineWidth',1.5,'LineStyle','-');
        text(kin_response_trig_index(plot_ind2)/raster_Fs,raster_ylim2+1,'Hit','Rotation',60,'FontSize',10);
        %text((kin_response_trig_index(plot_ind2)/raster_Fs)+0.25,raster_ylim2,'Reached','Rotation',60,'FontSize',8);
        hold on;
    end

    for plot_ind3 = 1:length(Move_onset_trig)
        if ((Move_onset_trig(plot_ind3,2) >= raster_lim1/raster_Fs) && (Move_onset_trig(plot_ind3,2) <= raster_lim2/raster_Fs))
            line([Move_onset_trig(plot_ind3,2), Move_onset_trig(plot_ind3,2)],[raster_ylim1,raster_ylim2],'Color','k','LineWidth',2,'LineStyle','--');
            text(Move_onset_trig(plot_ind3,2),raster_ylim2,'Move Onset','Rotation',60,'FontSize',9);
            %text(Move_onset_trig(plot_ind3,2)+0.25,raster_ylim2,'Onset','Rotation',60,'FontSize',8);
        end
    end
    axis([raster_lim1/raster_Fs raster_lim2/raster_Fs raster_ylim1 raster_ylim2]);
    set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85]);
    set(gca,'YTickLabel',{'F1','Fz','F2','FC1','FCz','FC2','C1','Cz','C2','CP1','CPz','CP2','C3','C4','Pos X ','Pos Y','Tang Velocity'},'FontSize',8);
    %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
    %set(gca,'XTickLabel',{'Target','Movement'});
    xlabel('Time (sec.)','FontSize',10);
    hold off;



%% Add Stimulus time correction for EEG latencies


    Move_onset_trig(:,2) = Move_onset_trig(:,2) + stimulus_time_correction;
    Target_cue_trig(:,2) = Target_cue_trig(:,2) + stimulus_time_correction;
    %events_filename = ['F:\\Nikunj_Data\\InMotion_Experiment_Data\\Subject_' Subject_name '\\Movement_events.txt'];
    events_filename = [folder_path 'Movement_events.txt'];      % Does not Work - 18-9-2013
    %save(events_filename,'Move_onset_trig','-ascii','-double');
    dlmwrite(events_filename,Move_onset_trig,'delimiter','\t','precision','%d %.4f');
    
    %Target_cue_filename = ['F:\\Nikunj_Data\\InMotion_Experiment_Data\\Subject_' Subject_name '\\Target_cue_events.txt'];
    Target_cue_filename = [folder_path 'Target_cue_events.txt'];
    %save(Target_cue_filename,'Target_cue_trig','-ascii','-double');
    dlmwrite(Target_cue_filename,Target_cue_trig,'delimiter','\t','precision','%d %.4f')

    % Append events
    % EEG = pop_importevent( EEG, 'event','F:\\Nikunj_Data\\InMotion_Experiment_Data\\Subject_FJ\\Move_events.txt','fields',{'type' 'latency'},'timeunit',1,'align',0);
    % EEG = eeg_checkset( EEG );
    %EEG = pop_importevent( EEG, 'event','F:\\Nikunj_Data\\InMotion_Experiment_Data\\Subject_FJ2\\Movement_events.txt','fields',{'type' 'latency'},'timeunit',1,'align',0);
    %EEG = pop_importevent( EEG, 'event','Move_onset_trig','fields',{'type' 'latency'},'timeunit',1,'align',0);
    EEG = pop_importevent(EEG);
    EEG = eeg_checkset(EEG);
    %EEG = pop_importevent( EEG, 'event','Target_cue_trig','fields',{'type' 'latency'},'timeunit',1,'align',0);
    EEG = pop_importevent(EEG);
    EEG = eeg_checkset(EEG);


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


% Save dataset
    if ~isempty(Block_num)
    EEG.setname = [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed_event_labels'];
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_preprocessed_event_labels.set'],'filepath',folder_path);    
    else
    EEG.setname = [Subject_name '_ses' num2str(Sess_num) '_preprocessed_event_labels'];        
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_preprocessed_event_labels.set'],'filepath',folder_path);
    end

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

end

%% Extract move epochs and rest epochs
move_trig_label = '100';%'S 32'; %'100'; % 'S 32'; %'S  8';
rest_trig_label = '200';%'S  2'; %'200'; % 'S  2';
epoch_dur = [-2.5 3.1];

if extract_epochs == 1
    
    % Extract move epochs
    if ~isempty(Block_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_preprocessed.set'], folder_path);
    else
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_preprocessed.set'], folder_path);
    end
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;        
       
    EEG = pop_epoch( EEG, {  move_trig_label  }, epoch_dur, 'newname', 'move_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    %EEG = pop_resample( EEG, 10);
    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_move_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    if ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_move_epochs.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_move_epochs.set'],'filepath',folder_path);
    end
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;


    % Extract rest epochs
    if ~isempty(Block_num)
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_preprocessed.set'], folder_path);
    else
        EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_preprocessed.set'], folder_path);
    end
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;        
       
    EEG = pop_epoch( EEG, {  rest_trig_label  }, epoch_dur, 'newname', 'move_epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    %EEG = pop_resample( EEG, 10);
    EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_rest_epochs'];
    EEG = eeg_checkset( EEG );
    
    % Save dataset
    if ~isempty(Block_num)
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_rest_epochs.set'],'filepath',folder_path);    
    else
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_rest_epochs.set'],'filepath',folder_path);
    end
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end

if train_LDA_model == 1
%% Calculate ERPs for Rest Epochs
baseline_int = [-2.5 -2.0];    

% Load SB_rest_epochs.set
if ~isempty(Block_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_rest_epochs.set'], folder_path);
else
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_rest_epochs.set'], folder_path);
end

Fs_eeg = EEG.srate;
rest_erp_time = EEG.xmin:1/Fs_eeg:EEG.xmax;

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
        % Subtract mean value of rest from rest trials
        rest_epochs(epoch_cnt,:,channel_cnt) = rest_epochs(epoch_cnt,:,channel_cnt) - rest_mean_baseline(channel_cnt,epoch_cnt);
    end
end


for channel_cnt = 1:no_channels
    rest_avg_channels(channel_cnt,:) =  mean(rest_epochs(:,:,channel_cnt));
    rest_std_channels(channel_cnt,:) =  std(rest_epochs(:,:,channel_cnt));
    rest_SE_channels(channel_cnt,:) = 1.96.*std(rest_epochs(:,:,channel_cnt))/sqrt(no_epochs);
end
% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
%% Calculate ERPs for Movement Epochs
% Load SB_move_epochs.set


if ~isempty(Block_num)
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_' spatial_filter_type '_move_epochs.set'], folder_path);
else
    EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_' spatial_filter_type '_move_epochs.set'], folder_path);
end

Fs_eeg = EEG.srate;
move_erp_time = EEG.xmin:1/Fs_eeg:EEG.xmax;

[no_channels,no_datapts,no_epochs] = size(EEG.data);
move_epochs = zeros([no_epochs,no_datapts,no_channels]); 
move_avg_channels = zeros(no_channels,no_datapts);
move_std_channels = zeros(no_channels,no_datapts);
move_SE_channels = [];

for epoch_cnt = 1:no_epochs
    for channel_cnt = 1:no_channels
        move_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt);
        move_mean_baseline(channel_cnt,epoch_cnt) = mean(move_epochs(epoch_cnt,find(move_erp_time == (baseline_int(1))):find(move_erp_time == (baseline_int(2))),channel_cnt));
        move_epochs(epoch_cnt,:,channel_cnt) = move_epochs(epoch_cnt,:,channel_cnt) - move_mean_baseline(channel_cnt,epoch_cnt);
    end
end

for channel_cnt = 1:no_channels
    move_avg_channels(channel_cnt,:) =  mean(move_epochs(:,:,channel_cnt));
    move_std_channels(channel_cnt,:) =  std(move_epochs(:,:,channel_cnt));
    move_SE_channels(channel_cnt,:) = 1.96.*std(move_epochs(:,:,channel_cnt))/sqrt(no_epochs);
end


% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
end
%% Plot ERPs
if plot_ERPs == 1
    figure; 
    %figure('units','normalized','outerposition',[0 0 1 1])
    hold on;
    plot_ind4 = 1;
    for RPind = 1:length(Channels_nos)
%         if plot_ind4 == 5                 % Commented by Nikunj on Oct 31,2013
%             plot_ind4 = plot_ind4 + 1;
%         end
        subplot(5,5,plot_ind4);hold on;
        plot(move_erp_time,move_avg_channels(Channels_nos(RPind),:),'b','LineWidth',2);
        jbfill(move_erp_time,move_avg_channels(Channels_nos(RPind),:)+ (move_SE_channels(Channels_nos(RPind),:)),...
           move_avg_channels(Channels_nos(RPind),:)- (move_SE_channels(Channels_nos(RPind),:)),'b','k',0,0.3);
       plot(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
       jbfill(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:)+ (rest_SE_channels(Channels_nos(RPind),:)),...
          rest_avg_channels(Channels_nos(RPind),:)- (rest_SE_channels(Channels_nos(RPind),:)),'r','k',0,0.3);
       
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),rest_time, rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,preprocessed_move_epochs(Channels_nos(RPind),:),'b',erp_time,standardize_move_epochs(Channels_nos(RPind),:),'k',erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        text(-1.5,2.5,EEG.chanlocs(Channels_nos(RPind)).labels,'Color','k');
        set(gca,'YDir','reverse');
        %if max(abs(move_avg_channels(Channels_nos(RPind),:))) <= 6
         if max((move_avg_channels(Channels_nos(RPind),:)+(move_SE_channels(Channels_nos(RPind),:)))) >= 6 || ...
                 min((move_avg_channels(Channels_nos(RPind),:)-(move_SE_channels(Channels_nos(RPind),:)))) <= -6
            axis([move_erp_time(1) move_erp_time(end) -15 15]);
            %axis([move_erp_time(1) 1 -6 6]);
            set(gca,'FontWeight','bold');            
        else
            axis([move_erp_time(1) move_erp_time(end) -6 6]);
            %axis([move_erp_time(1) 1 -15 15]);
            
        end
        line([0 0],[-30 30],'Color','k','LineWidth',2);  
        plot_ind4 = plot_ind4 + 1;
        grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
    
    end

    % subplot(4,3,5);
    % topoplot([],EEG.chanlocs,'style','blank','electrodes','labels','chaninfo',EEG.chaninfo);
    subplot(5,5,8);
    hold on;
    xlabel('\bfTime (sec.)', 'FontSize', 12);
    ylabel('\bfVoltage (\muV)','FontSize', 12);
    mtit(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'fontsize',14,'color',[0 0 0],'xoff',-.02,'yoff',.025);
    %title(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)));
    %legend('Average RP','Movement Onset','Orientation','Horizontal');
    %export_fig LG8_Average '-png' '-transparent';
    
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
%% LDA Classifier training & validation
if train_classifier == 1
% State features to be used
avg_sensitivity = [];
avg_specificity = [];
avg_accur       = [];
% num_channels    = [];
avg_TPR = [];
avg_FPR = [];

move_window = [-0.7 -0.1];
rest_window = [-0.7 -0.1];
classchannels = [48,14,49,9]; % C1, Cz, C2

mlim1 = abs(-2-(move_window(1)))*10+1;
mlim2 = abs(-2-(move_window(2)))*10+1;
rlim1 = abs(-1-(rest_window(1)))*10+1;
rlim2 = abs(-1-(rest_window(2)))*10+1;

num_diff = 0;               % calculate slope of EEG signals
gain = 10^num_diff;
if gain == 0
    gain = 1;
end
num_feature_per_chan = mlim2 - mlim1 + 1 - num_diff;
data_set = zeros(2*no_epochs,length(classchannels)*(length(mlim1:mlim2)-num_diff));
data_set_labels = zeros(2*no_epochs,1);

for train_ind = 1:no_epochs
    if num_diff == 0
        for chan_ind = 1:length(classchannels)
            ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
            ra2 = chan_ind*num_feature_per_chan;
            data_set(train_ind,ra1:ra2) = move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind));
        end
    else
        for chan_ind = 1:length(classchannels)
            ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
            ra2 = chan_ind*num_feature_per_chan;
            data_set(train_ind,ra1:ra2) = diff(move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind)),num_diff,2).*gain;
        end

    end
    data_set_labels(train_ind,1) = 1;
    
    if num_diff == 0               
       for chan_ind = 1:length(classchannels)
            ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
            ra2 = chan_ind*num_feature_per_chan;
            data_set(train_ind+no_epochs,ra1:ra2) = rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind));
       end        
    else 
        for chan_ind = 1:length(classchannels)
            ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
            ra2 = chan_ind*num_feature_per_chan;
            data_set(train_ind+no_epochs,ra1:ra2) = diff(rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind)),num_diff,2).*gain;
        end   
    end
    data_set_labels(train_ind+no_epochs,1) = 2;
end

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
 
%%
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
    sensitivity(cv_index) = CM(1,1)/(CM(1,1)+CM(1,2));
    specificity(cv_index) = CM(2,2)/(CM(2,2)+CM(2,1));
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

% Find Classifier coefficients with 'best' performance. 
performance = abs(sensitivity - specificity);
best_index = find(performance == min(performance));

% Recalculate coefficients for best performance classifier
trIdx = CVO.training(best_index);
teIdx = CVO.test(best_index);

[test_labels, Error, Posterior, LogP, Best_OutputCoefficients] = ...
       classify(data_set([teIdx;teIdx],:),data_set([trIdx;trIdx],:),data_set_labels([trIdx;trIdx],1), 'linear');

CM = confusionmat(data_set_labels([teIdx;teIdx],:),test_labels);
best_sensitivity = CM(1,1)/(CM(1,1)+CM(1,2));
best_specificity = CM(2,2)/(CM(2,2)+CM(2,1));
%%
% Write Coefficients to a text file
K12 = Best_OutputCoefficients(1,2).const;
L12 = Best_OutputCoefficients(1,2).linear;

% Find margins on classifier's decision
for len = 1:size(data_set,1)
    classifier_decision(len) = K12+data_set(len,:)*L12;
end

coeff_filename = [folder_path Subject_name '_coeffs.txt'];
fileid = fopen(coeff_filename,'w+');
fprintf(fileid,'**** Calibration Data for Subject %s *****\r\n',Subject_name);
fprintf(fileid,'Avg_Sensitivity = %.2f \r\n',avg_sensitivity*100);
fprintf(fileid,'Avg_Specificity = %.2f \r\n',avg_specificity*100);
fprintf(fileid,'Best_Sensitivity = %.2f \r\n',best_sensitivity*100);
fprintf(fileid,'Best_Specificity = %.2f \r\n',best_specificity*100);
fclose(fileid);
dlmwrite(coeff_filename,[K12;L12],'-append','delimiter','\t','precision','%.4f');

% Write Class Channels to a separate text file
channels_filename = [folder_path Subject_name '_channels.txt'];
dlmwrite(channels_filename,classchannels,'delimiter','\t','precision','%d');

end


%% Close-loop BCI testing in realtime. 
if test_LDA_model == 1
    
    [proc_eeg,num_move,num_rest,markers] = RDA(Subject_name,folder_path);
    markers = double(markers);
    markers(:,1) = markers(:,1) - markers(1,1);
    markers(:,1) = markers(:,1)*5/500;
    % Round upto 1 decimal place
    markers(:,1) = floor(markers(:,1)*10)/10;

%% Generate Raster Plot

    cloop_data = proc_eeg;       
    raster_Fs = 10;
    raster_lim1 = 1;
    raster_lim2 = length(cloop_data);
    raster_time = (0:1/raster_Fs:(length(cloop_data)-1)*(1/raster_Fs));
    %[no_channels,no_samples] = size(cloop_data);
    no_channels = length(Channels_nos);
    
    % Plot selected EEG channels and decision
    raster_data = zeros(length(cloop_data(1,raster_lim1:raster_lim2)),no_channels);
    for chan_index = 1:no_channels
        raster_data(:,chan_index) = cloop_data(Channels_nos(chan_index),:)';
    end
    raster_data = [raster_data num_move num_rest];
    
    % Standardization/Normalization of Signals; 
    % http://en.wikipedia.org/wiki/Standard_score
    % Standard score or Z-score is the number of standard deviations an
    % obervation is above/below the mean. 
    % z-score = (X - mu)/sigma;

    raster_zscore = zscore(raster_data);

    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 5;
    raster_ylim1 = 0;
    raster_ylim2 = (raster_col+1)*add_offset;
    figure;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
        plot(raster_time,raster_zscore(:,raster_index),myColors(raster_index));
        hold on;
    end
    
    axis([raster_lim1/raster_Fs raster_lim2/raster_Fs raster_ylim1 raster_ylim2]);
    set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70]);
    set(gca,'YTickLabel',{'Fz','FC1','FC2','Cz','CP1','CP2','F1','F2','C1','C2','CPz','Move','Rest','Tang Velocity'},'FontSize',8);
    %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
    %set(gca,'XTickLabel',{'Target','Movement'});
    xlabel('Time (sec.)','FontSize',10);
    hold off;

    %Plot Markers on the raster plot
     [num_markers, num_types] = size(markers);
    for plot_ind1 = 1:num_markers;
        if markers(plot_ind1,2) == 100
            line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','r');
        elseif markers(plot_ind1,2) == 200
            line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','g');
        end
    hold on;
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




        