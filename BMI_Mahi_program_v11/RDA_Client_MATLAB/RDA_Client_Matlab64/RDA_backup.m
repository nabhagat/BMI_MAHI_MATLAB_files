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
function RDA()

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
                        
                    case 4       % 32Bit Data block
                        % Read data and markers from message
                        [datahdr, data, markers] = ReadDataMessage(con, hdr, props);

                        % check tcpip buffer overflow
                        if lastBlock ~= -1 && datahdr.block > lastBlock + 1
                            disp(['******* Overflow with ' int2str(datahdr.block - lastBlock) ' blocks ******']);
                        end
                        lastBlock = datahdr.block;

                        % print marker info to MATLAB console
                        if datahdr.markerCount > 0
                            for m = 1:datahdr.markerCount
                                disp(markers(m));
                            end    
                        end

                        % Process EEG data, 
                        % in this case extract last recorded second,
                        EEGData = reshape(data, props.channelCount, length(data) / props.channelCount);
                        data1s = [data1s EEGData];
                        dims = size(data1s);
                        if dims(2) > 1000000 / props.samplingInterval
                            data1s = data1s(:, dims(2) - 1000000 / props.samplingInterval : dims(2));
                            avg = mean(mean(data1s.*data1s));
                            disp(['Average power: ' num2str(avg)]);

                            % set data buffer to empty for next full second
                            data1s = [];
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

