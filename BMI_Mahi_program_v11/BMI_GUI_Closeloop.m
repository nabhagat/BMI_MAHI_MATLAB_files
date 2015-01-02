% Close loop control program for GUI interface with BMI
% Developed by: Nikunj A. Bhagat, Graudate Student, University of Houston, 
% Non-Invasive Brain Machine Interface Systems Laboratory, 
% Date Created: July 8th, 2014
%--------------------------------------------------------------------------------------------------
%% Adapted from BMI_Mahi_Closeloop.m
%--------------------------------------------------------------------------------------------------
%% ***********Revisions 
% 5/22/14 - Computing spatial channel average (mean filtering) before
%           segmenting into sliding windows. 
%         - Major chnages in variables saved.
% 5/23/14 - Corrected formula for pkt_Size. *****Very Important
%         - 'sample_num' field added to 'marker' structure. This is used to calculate time (sec.) at which trigger occured.  
% 5/27/14 - 'S 10' Trigger used to start and stop closeloop predictions
% 6/29/14 - Changed resamp_Fs to 20 Hz, calculation of Mahalanobis distance
%           for sliding window
%--------------------------------------------------------------------------------------------------
%% ***********************************************************************   
% Main RDA Client function
function [processed_eeg, Overall_spatial_chan_avg, all_feature_vectors,GO_Probabilities, move_counts, marker_block, cloop_prob_threshold, cloop_cnts_threshold]...
                                                                                                                = BMI_GUI_Closeloop(CloopClassifier,serial_obj)
   
% Read Classifier parameters from file performance_new variable.
%classifier_coeffs = dlmread(coeff_filename,'\t',num_header_lines,0);
total_chnns = 32;           % change1
%classifier_channels = CloopClassifier.classchannels;
classifier_channels = [9 10 16 17];     % change2

resamp_Fs = 20; % required Sampling Freq
window_length = CloopClassifier.smart_window_length*resamp_Fs + 1; % No. of samples required, Earlier  = length(CloopClassifier.move_window(1):1/resamp_Fs:CloopClassifier.move_window(2));   
window_time = 1/resamp_Fs:1/resamp_Fs:window_length/resamp_Fs;

% Decide classifier with best accuracy
[max_acc_val,max_acc_index] = max(CloopClassifier.eeg_accur); 
Best_BMI_classifier = CloopClassifier.eeg_svm_model{max_acc_index};

% Classifier parameters
cloop_prob_threshold = CloopClassifier.opt_prob_threshold;         % change3     
cloop_cnts_threshold  = CloopClassifier.consecutive_cnts_threshold;        % Number of consecutive valid cnts required to accept as 'GO'

DEFINE_GO = 1;
DEFINE_NOGO = 2;
target_trig_label = 'S  8';
move_trig_label = 'S 16';  % 'S 32'; %'S  8'; %'100';
rest_trig_label = 'S  2';  % 'S  2'; %'200';
block_start_stop_label = 'S 10';

% Arduino Commands
%     ard = arduino('COM4');
%     ard.pinMode(30,'output');
%     ard.pinMode(32,'output');

% Change for individual recorder host
recorderip = '127.0.0.1';
% Establish connection to BrainVision Recorder Software 32Bit RDA-Port
% (use 51234 to connect with 16Bit Port)
con = pnet('tcpconnect', recorderip, 51244);
% Check established connection and display a message
stat = pnet(con,'status');
if stat > 0
disp('connection established');
end  

%% ------------------ Main reading loop ----------------------------------------
header_size = 24;
finish = false;
while ~finish
    try
        % check for existing data in socket buffer
        tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
        while ~isempty(tryheader)
            % Read header of RDA message
            hdr = ReadHeader(con);
            % Perform some action depending of the type of the data package
            switch hdr.type
                case 1       
                %% Initialize, Setup information like EEG properties
                    disp('Start');
                    % Read and display EEG properties
                    props = ReadStartMessage(con, hdr);
                    disp(props);
                    % Reset block counter to check overflows
                    lastBlock = -1;
                    % set data buffer to empty
                    dataX00ms = [];

                    %********** Initialize variables
                    orig_Fs = (1000000/props.samplingInterval);                     % Original Sampling Frequency
                    downsamp_factor = orig_Fs/resamp_Fs;
                    % pkt_size = (1000000 / props.samplingInterval)/2;                 % samplingInterval = 2000
                    pkt_size =  int32((window_length/resamp_Fs)*orig_Fs);     % half of window_size because of prop.samplingInterval
                    filt_order = 4;
                    n_channels = props.channelCount;
                    new_pkt = zeros(n_channels,pkt_size);
                    prev_pkt = zeros(n_channels,pkt_size);
                    hp_filt_pkt = zeros(n_channels,pkt_size);
                    lp_filt_pkt = zeros(length(classifier_channels),pkt_size);      % Use only classifier channels
                    spatial_filt_pkt = zeros(n_channels,pkt_size);
                    
                    firstblock = true;          % First block of data read in 
                    marker_block = [];
                    marker_type = [];
                    
                    processed_eeg = [];
                    %unprocessed_eeg = [];
                    Overall_spatial_chan_avg = [];
                    %lpf_eeg = [];
                    %raster_signal = [];
                    all_feature_vectors = [];
                    GO_Probabilities = [];
                    valid_move_cnt = 0;
                    move_counts = [];
                    
                    feature_matrix = zeros(length(classifier_channels),pkt_size/downsamp_factor);
                    sliding_window = zeros(1,window_length); %zeros(length(classifier_channels),window_length);
                    %move_command = [];           % Number of predicted moves
                    %rest_command = [];           % Number of predicted rests
                    %likert_score = [];          % To be used for recording trial success/ failure
                    %prediction = [];
                    start_prediction = false;
                    kcount = 0;


        %                         baseline = feature_data;     % Initialize           
        %                         new_baseline = false;
        %                         get_new_baseline = true;
        %                         false_starts = 0;
        %                         num_starts = 0;

                    % Create Common Avg Ref matrix of size 64x64
                   % total_chnns = 68;
                    car_chnns_eliminate = [];
                    num_car_channels = total_chnns - length(car_chnns_eliminate);          % Number of channels to use for CAR
                    M_CAR =  (1/num_car_channels)*(diag((num_car_channels-1)*ones(total_chnns,1)) - (ones(total_chnns,total_chnns)-diag(ones(total_chnns,1))));
                    for elim = 1:length(car_chnns_eliminate)
                        M_CAR(:,car_chnns_eliminate(elim)) = 0;
                    end

                    % Create Large Laplacian Matrix - Need to revise
                    if total_chnns == 64
                    Neighbors = [
                                 38, 1, 37, 48, 39;
                                  5, 28, 4, 14, 6;
                                 39, 2, 38, 49, 40;
                                  9, 34, 8, 19, 10;
                                 32, 28, 43, 53, 44;
                                 10, 35, 9, 11, 20;
                                 48, 38, 47, 57, 49;
                                 14, 5, 13, 25, 15;
                                 49, 39, 48, 58, 50;
                                 19, 9, 18, 61, 20; 
                                 53, 32, 52, 62, 54;
                                 20, 10, 19, 63, 21     
                                 ];
                    elseif total_chnns == 32
                    % For modified 32 channels          % change4
                    % Classifier channels = [9 10 16 17];
                    Neighbors = [
                                  9,  1,  7, 23, 11;  
                                 10,  2,  8, 24, 12; 
                                 11,  3,  9, 25, 13;  
                                 16,  4, 14, 27, 18;  
                                 17,  5, 15, 28, 19; 
                                 18,  6, 16, 29, 20;
                                 23,  9, 21, 30, 25;
                                 24, 10, 22, 31, 26;
                                 25, 11, 23, 32, 26;    
                                 ];  
                    else
                        error('Total number of EEG channels mismatch. Kindly Check!!');
                    end
                             
                    required_chnns = unique(Neighbors(:));
                    L_LAP = eye(total_chnns);

                    for nn_row = 1:size(Neighbors,1) 
                        L_LAP(Neighbors(nn_row,1),Neighbors(nn_row,2:end)) = -0.25;
                    end

                case 4  
        %% Process EEG, create feature vector, SVM classify, 32Bit Data block
                    % Read data and markers from message
                    [datahdr, data, markers] = ReadDataMessage(con, hdr, props);
                    if firstblock == true
                        marker_block = [marker_block; [datahdr.block,1]];
                        firstblock = false;
                    end
                    % check tcpip buffer overflow
                    if lastBlock ~= -1 && datahdr.block > lastBlock + 1
                        disp(['******* Overflow with ' int2str(datahdr.block - lastBlock) ' blocks ******']);
                    end
                    lastBlock = datahdr.block;

                    % print marker info to MATLAB console
                    if datahdr.markerCount > 0                            
                        for m = 1:datahdr.markerCount
                            %disp(markers(m));
                            if (strcmp(markers(m).description,move_trig_label))         % Movement onset
                                %start_prediction = false;
                                marker_block = [marker_block; [markers(m).sample_num,100]];
                                %kcount = kcount + 1;
                            elseif (strcmp(markers(m).description,rest_trig_label))     % Targets appear
                               % baseline = feature_data;  
                               % new_baseline = true;
                               marker_block = [marker_block; [markers(m).sample_num,200]];
                            elseif (strcmp(markers(m).description,block_start_stop_label))     % Start/Stop Prediction
                               start_prediction = ~start_prediction; 
                               marker_block = [marker_block; [markers(m).sample_num,50]];
                               if start_prediction == true
                                   disp('Starting Prediction');
                               else
                                   disp('Stoping Prediction');
                               end
                            end                                   
                        end                          
                    end

                           
                    % Process raw EEG data, 
                    EEGData = reshape(data, props.channelCount, length(data) / props.channelCount);
                    EEGData = double(EEGData);  % Very Important!! Always convert to double precision
                    EEGData = EEGData.*props.resolutions(1,1);
                    dataX00ms = [dataX00ms EEGData];
                    dims = size(dataX00ms);                  
                    % For 700 msec windows, % For 500 msec windows: if dims(2) > (1000000 / props.samplingInterval)/2 
                    if dims(2) > pkt_size     
                       new_pkt = dataX00ms(:,1:pkt_size);                      % Select 64xN size of data packet i.e 2*N msec
                       dataX00ms = dataX00ms(:,pkt_size+1 : dims(2));          % Move all excess data to next data packet        

                       % High pass filter 0.1 Hz
                       for no_chns = 1:length(required_chnns)
                           hp_filt_pkt(required_chnns(no_chns),:) = pkt_hp_filter(new_pkt(required_chnns(no_chns),:),...
                               prev_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size),...
                               hp_filt_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size),...
                               orig_Fs);
                       end

                       % Re-reference the data                                                     
                       %spatial_filt_pkt = M_CAR*hp_filt_pkt;      % Common Average referencing  
                       spatial_filt_pkt = L_LAP*hp_filt_pkt;          % Large Laplacian filtering

                       % Low pass filter 1 Hz
                       for no_chns = 1:length(classifier_channels)                                
                           lp_filt_pkt(no_chns,:) = pkt_lp_filter(spatial_filt_pkt(classifier_channels(no_chns),:),...
                               prev_pkt(classifier_channels(no_chns),(pkt_size - filt_order)+1:pkt_size),...
                               lp_filt_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size),...
                               orig_Fs);
                       end

                       % Downsample data to 10 Hz
                       %Filtered_EEG_Downsamp = downsample(lp_filt_pkt',downsamp_factor)' - repmat(mean(baseline,2),1,window_size);
                       Filtered_EEG_Downsamp = downsample(lp_filt_pkt',downsamp_factor)';        % No baseline correction
                       % 1. Compute spatial channel average (mean filtering)
                       spatial_chan_avg = mean(Filtered_EEG_Downsamp,1);                       
                       
  if start_prediction == true                           
                       % Implement Sliding Window, calculate feature vectors 
                       for sl_index = 1:window_length
                           sliding_window = [sliding_window(1,2:window_length) spatial_chan_avg(1,sl_index)];
                           % Create feature vector for SVM classification
                 % 2. Features - slope, -ve peak, AUC and Mahalanobis distance (in this order only) 
                          feature_vector = [(sliding_window(end) - sliding_window(1))/(window_time(end) - window_time(:,1));
                                             min(sliding_window,[],2);
                                             trapz(window_time,sliding_window);
                                             sqrt((sliding_window-CloopClassifier.smart_Mu_move)/(CloopClassifier.smart_Cov_Mat)*(sliding_window-CloopClassifier.smart_Mu_move)')];
                                             %sqrt(mahal(sliding_window,CloopClassifier.move_ch_avg(:,:)))];
                         all_feature_vectors = [ all_feature_vectors feature_vector];
                 % 3. Make a prediction; Class 1: Movement; Class 2: Rest
                         [Decision_slid_win(sl_index),Acc_slid_win,Prob_slid_win(sl_index,:)] = svmpredict(DEFINE_NOGO,feature_vector',Best_BMI_classifier,'-b 1 -q');  %#ok<ASGLU>
                         GO_Probabilities = [GO_Probabilities; Prob_slid_win(sl_index,1)];
                         
                         %if (Decision_slid_win(sl_index) == DEFINE_GO) || (Prob_slid_win(sl_index,1) > prob_est_margin)        % param 1
                         if (Prob_slid_win(sl_index,1) > cloop_prob_threshold)
                                valid_move_cnt = valid_move_cnt + 1;
        %                                     ard.digitalWrite(30,1);
        %                                     pause(0.01);
        %                                     ard.digitalWrite(30,0);
                                if valid_move_cnt >= cloop_cnts_threshold                            % param 2
                    
                               %****check serial port to decide if return command
        % %                                     serial_read = fread(serial_obj, 1, 'uchar');
        % % % %                                    while (serial_read ~= 'm')
        % % % %                                        serial_read = fread(serial_obj, 1, 'uchar');
        % % % %                                        display(serial_read)
        % % % %                                    end                                     
        % %                                     if  serial_read == 'm'
        % %                                              display(serial_read);

        %                                              ard.digitalWrite(32,1);
        %                                              pause(0.05);
        %                                              ard.digitalWrite(32,0);
        %                                              
                                            %fwrite(serial_obj,[7]);    % change5            
                                            disp('--->GO');
                                            marker_block = [marker_block; [datahdr.block*datahdr.points,300]];      % Not completely accurate

        %                                             disp('Do you think that it was you that made the cursor move?');
        %                                             likert_score = [ likert_score; input('Enter Score: ')];      
                                            
                                end % endif
                               %move_command = [move_command; 1];
                               %rest_command = [rest_command; 0];
                         else
                               %move_command = [move_command; 0];
                               %rest_command = [rest_command; 1];
                               valid_move_cnt = 0;           
                               fprintf('.');
                               kcount = kcount+1; 
                               if kcount >= 15;
                                   kcount = 0;
                                   fprintf('\n');
                               end
                         end % endif
%                                raster_signal = [raster_signal; [Decision_slid_win, valid_move_cnt]];
                                move_counts = [move_counts valid_move_cnt];
                                if  valid_move_cnt >= cloop_cnts_threshold;
                                     valid_move_cnt = 0;
                                end
                                
                       end % end for sl_index = 1:window_length
                       
   end % end if start_prediction == true
                        % Miscellaneous
                        prev_pkt = new_pkt;
    %                     if new_baseline == true
    %                         processed_eeg = [processed_eeg baseline];
    %                         new_baseline = false;
    %                     end
%                        unprocessed_eeg = [unprocessed_eeg new_pkt];
                        processed_eeg   = [processed_eeg Filtered_EEG_Downsamp];
                        Overall_spatial_chan_avg = [Overall_spatial_chan_avg spatial_chan_avg];
%                         lpf_eeg = [lpf_eeg lp_filt_pkt];
                       
                    end    
        
                case 3       
        %% Stop message   
                        disp('Stop');
                        data = pnet(con, 'read', hdr.size - header_size);
                        finish = true;

                otherwise
        %% ignore all unknown types, but read the package from buffer
                        data = pnet(con, 'read', hdr.size - header_size);
            end % end Switch
        tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
        end % end while ~isempty(tryheader)
    catch err
        disp(err.message);
    end % end try/catch
end % end Main loop

% Close all open socket connections
pnet('closeall');

% Display a message
disp('connection closed');
%delete(ard);

    
%% Generate Raster Plot
% % 
% %     myColors = ['g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];
% %     cloop_data = [move_command, (-1)*rest_command, raster_signal(:,2),raster_signal(:,1)];       
% %     raster_Fs = 10;
% %     raster_lim1 = 1;
% %     raster_lim2 = size(cloop_data,1);
% %     raster_time = (0:1/raster_Fs:(length(cloop_data)-1)*(1/raster_Fs));
% % % %     %[no_channels,no_samples] = size(cloop_data);
% % % %     no_channels = length(Channels_nos);
% % % %     
% % % %     % Plot selected EEG channels and decision
% % % %     raster_data = zeros(length(cloop_data(1,raster_lim1:raster_lim2)),no_channels);
% % % %     for chan_index = 1:no_channels
% % % %         raster_data(:,chan_index) = cloop_data(Channels_nos(chan_index),:)';
% % % %     end
% % % %     raster_data = [raster_data num_move num_rest];
% %     raster_data = cloop_data;
% %     
% %     % Standardization/Normalization of Signals; 
% %     % http://en.wikipedia.org/wiki/Standard_score
% %     % Standard score or Z-score is the number of standard deviations an
% %     % obervation is above/below the mean. 
% %     % z-score = (X - mu)/sigma;
% % 
% %     raster_zscore = zscore(raster_data);
% %     
% % 
% %     % Plot the rasters; Adjust parameters for plot
% %     [raster_row,raster_col] = size(raster_zscore);
% %     add_offset = 5;
% %     raster_ylim1 = 0;
% %     raster_ylim2 = (raster_col+1)*add_offset;
% %     figure;
% %     for raster_index = 1:raster_col;
% %         raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
% %         plot(raster_time,raster_zscore(:,raster_index),myColors(raster_index));
% %         hold on;
% %     end
% %     
% %     axis([raster_lim1/raster_Fs raster_lim2/raster_Fs raster_ylim1 raster_ylim2]);
% % %     set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70]);
% % %     set(gca,'YTickLabel',{'Fz','FC1','FC2','Cz','CP1','CP2','F1','F2','C1','C2','CPz','Move','Rest','Tang Velocity'},'FontSize',8);
% %     %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
% %     %set(gca,'XTickLabel',{'Target','Movement'});
% %     xlabel('Time (sec.)','FontSize',10);
% %     hold off;
% % 
% %     markers = double(marker_block);
% %     markers(:,1) = markers(:,1) - markers(1,1);
% %     markers(:,1) = markers(:,1)*double(datahdr.points)/orig_Fs;
% %     % Round upto 1 decimal place
% %     markers(:,1) = floor(markers(:,1)*10)/10;
% %     
% %     %Plot Markers on the raster plot
% %      [num_markers, num_types] = size(markers);
% %     for plot_ind1 = 1:num_markers;
% %         if markers(plot_ind1,2) == 100
% %             line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','r');
% %         elseif markers(plot_ind1,2) == 200
% %             line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','g');
% %         end
% %     hold on;
% %     end
% %     
% %     for plot_ind2 = 1:length(raster_signal)
% %         if raster_signal(plot_ind2,2) == num_valid_cnts
% %             line([raster_time(plot_ind2) raster_time(plot_ind2)], [0 100],'Color','k');
% %         end
% %     end
    
%% ***********************************************************************
% Read the message header
function hdr = ReadHeader(con)
    % con    tcpip connection object
    
    % define a struct for the header
    hdr = struct('uid',[],'size',[],'type',[]);

    % read id, size and type of the message
    % swapbytes is important for correct byte order of MATLAB variables
    % pnet behaves somehow strange with byte order option
    hdr.uid = pnet(con,'read', 16);
    hdr.size = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
    hdr.type = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));


%% ***********************************************************************   
% Read the start message
function props = ReadStartMessage(con, hdr)
    % con    tcpip connection object    
    % hdr    message header
    % props  returned eeg properties

    % define a struct for the EEG properties
    props = struct('channelCount',[],'samplingInterval',[],'resolutions',[],'channelNames',[]);

    % read EEG properties
    props.channelCount = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
    props.samplingInterval = swapbytes(pnet(con,'read', 1, 'double', 'network'));
    props.resolutions = swapbytes(pnet(con,'read', props.channelCount, 'double', 'network'));
    allChannelNames = pnet(con,'read', hdr.size - 36 - props.channelCount * 8);
    props.channelNames = SplitChannelNames(allChannelNames);

    
%% ***********************************************************************   
% Read a data message
function [datahdr, data, markers] = ReadDataMessage(con, hdr, props)
    % con       tcpip connection object    
    % hdr       message header
    % props     eeg properties
    % datahdr   data header with information on datalength and number of markers
    % data      data as one dimensional arry
    % markers   markers as array of marker structs
    
    % Define data header struct and read data header
    datahdr = struct('block',[],'points',[],'markerCount',[]);

    datahdr.block = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
    datahdr.points = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
    datahdr.markerCount = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));

    % Read data in float format
    data = swapbytes(pnet(con,'read', props.channelCount * datahdr.points, 'single', 'network'));

    % Define markers struct and read markers
    markers = struct('size',[],'position',[],'points',[],'channel',[],'type',[],'description',[],'sample_num',[]);
    for m = 1:datahdr.markerCount
        marker = struct('size',[],'position',[],'points',[],'channel',[],'type',[],'description',[]);

        % Read integer information of markers
        marker.size = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.position = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.points = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.channel = swapbytes(pnet(con,'read', 1, 'int32', 'network'));
        marker.sample_num = ((datahdr.block - 1)*datahdr.points + marker.position);       % delayed by 110 msec as compared to EEGLAB. Why?
                
        % type and description of markers are zero-terminated char arrays
        % of unknown length
        c = pnet(con,'read', 1);
        while c ~= 0
            marker.type = [marker.type c];
            c = pnet(con,'read', 1);
        end

        c = pnet(con,'read', 1);
        while c ~= 0
            marker.description = [marker.description c];
            c = pnet(con,'read', 1);
        end
        
        % Add marker to array
        markers(m) = marker;  
    end

    
%% ***********************************************************************   
% Helper function for channel name splitting, used by function
% ReadStartMessage for extraction of channel names
function channelNames = SplitChannelNames(allChannelNames)
    % allChannelNames   all channel names together in an array of char
    % channelNames      channel names splitted in a cell array of strings

    % cell array to return
    channelNames = {};
    
    % helper for actual name in loop
    name = [];
    
    % loop over all chars in array
    for i = 1:length(allChannelNames)
        if allChannelNames(i) ~= 0
            % if not a terminating zero, add char to actual name
            name = [name allChannelNames(i)];
        else
            % add name to cell array and clear helper for reading next name
            channelNames = [channelNames {name}];
            name = [];
        end
    end

%% Filter raw eeg signals
function filtered_data = filter_raw_data(new_data,prev_data)

    window_length = 5;
    append_data = [prev_data(:,(length(prev_data)-(window_length-1)+1):length(prev_data)) new_data(:,:)];
    filtered_data = zeros(1,length(new_data));
    for fil_ind = window_length:length(append_data)
        filtered_data(fil_ind) = (1/window_length)*(append_data(fil_ind) + append_data(fil_ind-1) + append_data(fil_ind-2) + append_data(fil_ind-3) + append_data(fil_ind-4));
    end

filtered_data = filtered_data(:,5:length(append_data));        
