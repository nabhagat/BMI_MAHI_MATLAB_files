%% ***********************************************************************   
% Simple MATLAB RDA Client
%
% Demonstration file for implementing a simple MATLAB client for the
% RDA tcpip interface of the BrainVision Recorder.
% It reads all information of the recorded EEG,
% prints EEG and marker information to the
% MATLAB console and calculates and prints the average power every second.
%
%
% Brain Products GmbH 
% Gilching/Freiburg, Germany
% www.brainproducts.com
%
%
% This RDA Client uses version 2.x of the tcp/udp/ip Toolbox by
% Peter Rydesäter which can be downloaded from the Mathworks website
%
% 

%% ***********************************************************************   
% Main RDA Client function
function [unprocessed_eeg, raster_signal,move_command,rest_command,marker_block,likert_score] = RDA(subj_name,filepath,serial_obj)

    % Folder Path & Subject name
    Subject_name = subj_name;
    folder_path = filepath;
    coeff_filename = [folder_path Subject_name '_coeffs.txt'];
    channels_filename = [folder_path Subject_name '_channels.txt'];
    num_header_lines = 5;
    
    % Read Classifier coefficient & Channels Numbers from file.
    classifier_coeffs = dlmread(coeff_filename,'\t',num_header_lines,0);
    classifier_channels = dlmread(channels_filename,'\t');
    window_size = (length(classifier_coeffs)-1)/length(classifier_channels);        % 5*100ms = 500 ms, Number of feature per channel
    
    
    % Classifier parameters
    decision_margin = 2;
    num_valid_cnts  = 4;
    likert_score = [];
    
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
   
    
    
    % --- Main reading loop ---
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
                    case 1       % Start, Setup information like EEG properties
                        disp('Start');
                        % Read and display EEG properties
                        props = ReadStartMessage(con, hdr);
                        disp(props);

                        % Reset block counter to check overflows
                        lastBlock = -1;

                        % set data buffer to empty
                        data1s = [];
                        data500ms = [];
                        % pkt_size = (1000000 / props.samplingInterval)/2;                 % samplingInterval = 2000
                        pkt_size =  int32((window_size*100)/2);     % half of window_size becuase of prop.samplingInterval
                         filt_order = 4;
                        n_channels = props.channelCount;
                        new_pkt = zeros(n_channels,pkt_size);
                        prev_pkt = zeros(n_channels,pkt_size);
                        hp_filt_pkt = zeros(n_channels,pkt_size);
                        lp_filt_pkt = zeros(length(classifier_channels),pkt_size);                                % Use only classifier channels
                        spatial_filt_pkt = zeros(n_channels,pkt_size);
                        
                        
                        orig_Fs = (1000000 / props.samplingInterval);   % Original Sampling Frequency
                        resamp_Fs = 10;                                 % required Sampling Freq
                        downsamp_factor = orig_Fs/resamp_Fs;
                        %downsamp_pkt = zeros(length(classifier_channels),pkt_size/downsamp_factor);
                        
                        feature_data = zeros(length(classifier_channels),pkt_size/downsamp_factor);
                        baseline = feature_data;                             % Initialize           
                        sliding_window = zeros(length(classifier_channels),window_size);
                        move_command = [];           % Number of predicted moves
                        rest_command = [];           % Number of predicted rests
                        valid_move_cnt = 0;
                        processed_eeg = [];
                        unprocessed_eeg = [];
                        raster_signal = [];
                        marker_block = [];
                        marker_type = [];
                        new_baseline = false;
                        get_new_baseline = true;
                        start_prediction = false;
                        false_starts = 0;
                        num_starts = 0;
                        firstblock = true;
                        prediction = [];
                        
                        
                        % Create Common Avg Ref matrix of size 64x64
                        total_chnns = 64;
                        car_chnns_eliminate = [];
                        num_car_channels = total_chnns - length(car_chnns_eliminate);          % Number of channels to use for CAR
                        M_CAR =  (1/num_car_channels)*(diag((num_car_channels-1)*ones(total_chnns,1)) - (ones(total_chnns,total_chnns)-diag(ones(total_chnns,1))));
                        for elim = 1:length(car_chnns_eliminate)
                            M_CAR(:,car_chnns_eliminate(elim)) = 0;
                        end
                        
                        % Create Large Laplacian Matrix
                        Neighbors = [ 38, 1, 37, 48, 39;
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
                                             20, 10, 19, 63, 21];
                        required_chnns = unique(Neighbors(:));
                        L_LAP = eye(total_chnns);
                        
                        for nn_row = 1:size(Neighbors,1) 
                            L_LAP(Neighbors(nn_row,1),Neighbors(nn_row,2:end)) = -0.25;
                        end
                        
                        
                    case 4       % 32Bit Data block
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
                                if (strcmp(markers(m).description,'R128'))
                                    marker_block = [marker_block; [datahdr.block,100]];
                                elseif (strcmp(markers(m).description,'S  2'))
                                     baseline = feature_data;  
                                    new_baseline = true;
                                    marker_block = [marker_block; [datahdr.block,200]];
                                end                                   
                            end    
                            
                            if datahdr.markerCount == 2
                                num_starts = num_starts + 1;
                                
                                if num_starts == false_starts +1
                                    start_prediction = true;
                                    disp('Starting classifier ......');
                                    
                                else
                                    start_prediction = false;
                                    disp('Stop classifier ......');
                                end                                 
                                    
                            end                          
                        end

if start_prediction == true                        
                        % Process EEG data, 
                        % in this case extract last recorded second,
                        EEGData = reshape(data, props.channelCount, length(data) / props.channelCount);
                        EEGData = double(EEGData);  % Very Important!! Always convert to double precision
                        EEGData = EEGData.*props.resolutions(1,1);
                        %data1s = [data1s EEGData];
                        data500ms = [data500ms EEGData];
                        %dims = size(data1s);
                        dims = size(data500ms);
                        
                        % For 500 msec windows
                        % if dims(2) > (1000000 / props.samplingInterval)/2
                        
                        % For 700 msec windows
                        if dims(2) > pkt_size
       
%                             data1s = data1s(:, dims(2) - 1000000 / props.samplingInterval : dims(2));
%                             avg = mean(mean(data1s.*data1s));
%                             disp(['Average power: ' num2str(avg)]);
                            % set data buffer to empty for next full second
%                            data1s = [];
                            new_pkt = data500ms(:,1:pkt_size);                      % Select 64xN size of data packet i.e 2*N msec
                            data500ms = data500ms(:,pkt_size+1 : dims(2));   % Move all excess data to next data packet        
                            %data500ms = [];
                            
                            % High pass filter 0.1 Hz
                            for no_chns = 1:length(required_chnns)
                                hp_filt_pkt(required_chnns(no_chns),:) = pkt_hp_filter(new_pkt(required_chnns(no_chns),:),prev_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size),hp_filt_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size));
                            end
                            
                            % Re-reference the data                                                     
%                             avg_ref = mean(hp_filt_pkt,1);
%                             for no_chns = 1:n_channels
%                                  car(no_chns,:) = hp_filt_pkt(no_chns,:) - avg_ref;
%                                  %car(no_chns,:) = hp_filt_pkt(no_chns,:);
%                             end

                                %spatial_filt_pkt = M_CAR*hp_filt_pkt;      % Common Average referencing 
                                
                                spatial_filt_pkt = L_LAP*hp_filt_pkt;          % Large Laplacian filtering
                                

                                % Low pass filter 1 Hz
                            for no_chns = 1:length(classifier_channels)                                
                                lp_filt_pkt(no_chns,:) = pkt_lp_filter(spatial_filt_pkt(classifier_channels(no_chns),:),prev_pkt(classifier_channels(no_chns),(pkt_size - filt_order)+1:pkt_size),lp_filt_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size));
                            end
                            
                            % Downsample data to 10 Hz
%                             for no_chns = 1:n_channels
%                                 downsamp_pkt(no_chns,:) = downsample(lp_filt_pkt(no_chns,:),downsamp_factor);
%                             end

% Added 1/15/2014 - Nikunj; Get new baseline now
%                             if get_new_baseline == true
%                                 baseline = downsample(lp_filt_pkt',downsamp_factor)';
%                                 get_new_baseline = false;
%                             end
                            
                            feature_data = downsample(lp_filt_pkt',downsamp_factor)' - repmat(mean(baseline,2),1,window_size);

                            
                            % Select only required channels
%                            feature_data = downsamp_pkt(classifier_channels,:);
                  
                            % Implement Sliding Window & make prediction
                            curr_window = feature_data;
                            for sl_index = 1:window_size
                                sliding_window = [sliding_window(:,2:window_size) curr_window(:,sl_index)];
                                % Make a prediction
                                % Class 1: Movement
                                % Class 2: Rest
                                feature_vector = sliding_window';
                                feature_vector = [1;feature_vector(:)];
                                classifier_decision = feature_vector'*classifier_coeffs;  
                                
                                if abs(classifier_decision) >= 1000
                                    classifier_decision = 0;
                                end
                                
                                if classifier_decision > decision_margin                           % param 1
                                    valid_move_cnt = valid_move_cnt + 1;
%                                     ard.digitalWrite(30,1);
%                                     pause(0.01);
%                                     ard.digitalWrite(30,0);
                                    if valid_move_cnt >= num_valid_cnts                            % param 2
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
%                                             fwrite(serial_obj,'1');
                                                disp('*************Sending Move command');
                                             
%                                             disp('Do you think that it was you that made the cursor move?');
%                                             likert_score = [ likert_score; input('Enter Score: ')];
                                             

% %                                     end

                                    
%                                    else
%                                         move_command = [move_command; 0];
%                                         rest_command = [rest_command; 1];
                                    end
                                    move_command = [move_command; 1];
                                    rest_command = [rest_command; 0];
                                    
                                elseif classifier_decision < -decision_margin
                                    move_command = [move_command; 0];
                                    rest_command = [rest_command; 1];
                                    valid_move_cnt = 0;           
                                    disp('Rest');
                                else
                                    move_command = [move_command; 0];
                                    rest_command = [rest_command; 0];
                                    valid_move_cnt = 0;
                                end
                               
                                raster_signal = [raster_signal; [classifier_decision, valid_move_cnt]];
                                
                                if  valid_move_cnt >= num_valid_cnts;
                                     valid_move_cnt = 0;
                                end
                                
                            end

                            % Miscellaneous
                            prev_pkt = new_pkt;
                            if new_baseline == true
                                processed_eeg = [processed_eeg baseline];
                                new_baseline = false;
                            end
                            % processed_eeg = [processed_eeg spatial_filt_pkt];
%                             processed_eeg = [processed_eeg feature_vector];
                            unprocessed_eeg = [unprocessed_eeg new_pkt];
                        end

end
                    case 3       % Stop message   
                        disp('Stop');
                        data = pnet(con, 'read', hdr.size - header_size);
                        finish = true;

                    otherwise    % ignore all unknown types, but read the package from buffer 
                        data = pnet(con, 'read', hdr.size - header_size);
                end
                tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
            end
        catch
            er = lasterror;
            disp(er.message);
        end
    end % Main loop
    
    % Close all open socket connections
    pnet('closeall');
    
    % Display a message
    disp('connection closed');
    %delete(ard);
    
    
%% Generate Raster Plot

    myColors = ['g','r','b','k','y','c','m','g','r','b','k','b','r','m','g','r','b','k','y','c','m'];
    cloop_data = [move_command, (-1)*rest_command, raster_signal(:,2),raster_signal(:,1)];       
    raster_Fs = 10;
    raster_lim1 = 1;
    raster_lim2 = size(cloop_data,1);
    raster_time = (0:1/raster_Fs:(length(cloop_data)-1)*(1/raster_Fs));
% %     %[no_channels,no_samples] = size(cloop_data);
% %     no_channels = length(Channels_nos);
% %     
% %     % Plot selected EEG channels and decision
% %     raster_data = zeros(length(cloop_data(1,raster_lim1:raster_lim2)),no_channels);
% %     for chan_index = 1:no_channels
% %         raster_data(:,chan_index) = cloop_data(Channels_nos(chan_index),:)';
% %     end
% %     raster_data = [raster_data num_move num_rest];
    raster_data = cloop_data;
    
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
%     set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70]);
%     set(gca,'YTickLabel',{'Fz','FC1','FC2','Cz','CP1','CP2','F1','F2','C1','C2','CPz','Move','Rest','Tang Velocity'},'FontSize',8);
    %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
    %set(gca,'XTickLabel',{'Target','Movement'});
    xlabel('Time (sec.)','FontSize',10);
    hold off;

    markers = double(marker_block);
    markers(:,1) = markers(:,1) - markers(1,1);
    markers(:,1) = markers(:,1)*double(datahdr.points)/orig_Fs;
    % Round upto 1 decimal place
    markers(:,1) = floor(markers(:,1)*10)/10;
    
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
    
    for plot_ind2 = 1:length(raster_signal)
        if raster_signal(plot_ind2,2) == num_valid_cnts
            line([raster_time(plot_ind2) raster_time(plot_ind2)], [0 100],'Color','k');
        end
    end
    

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
    markers = struct('size',[],'position',[],'points',[],'channel',[],'type',[],'description',[]);
    for m = 1:datahdr.markerCount
        marker = struct('size',[],'position',[],'points',[],'channel',[],'type',[],'description',[]);

        % Read integer information of markers
        marker.size = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.position = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.points = swapbytes(pnet(con,'read', 1, 'uint32', 'network'));
        marker.channel = swapbytes(pnet(con,'read', 1, 'int32', 'network'));

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
   



