%% ***********************************************************************   
% MATLAB RDA Client with GUI
%
% Demonstration file for implementing a simple MATLAB client for the
% RDA tcpip interface of the BrainVision Recorder.
% It reads all information of the recorded EEG, displays the first channel
% and its Fourier transform and prints EEG and marker information to the
% MATLAB console.
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
function RDAGUI()

    % global definitions of controls for easy use in other functions
    global editHost;        % Host Name or IP edit control
    global axTime;          % Time domain axes
    global axFreq;          % Frequency domain axes

    % Create and hide the GUI figure as it is being constructed.
    f = figure('Visible','off',...
               'CloseRequestFcn',{@RDA_CloseRequestFcn},...
               'Position',[0,0,640,480]);
   
    % *** Construct the controls ***
    % controls for connection with host
    lblHost = uicontrol('Style','text','String','Host:',...
                        'BackgroundColor',get(f,'Color'),'Position',[25,448,50,16]);
    editHost = uicontrol('Style','edit','String','127.0.0.1',...
                         'Position',[75,450,150,16]);
    btConnect = uicontrol('Style','pushbutton','String','Connect',...
                          'Position',[230,450,100,16],...
                          'Callback',{@btConnect_Callback});
                      
    % construct the axes to display time and frequency domain data
    axTime = axes('Units','Pixels','Position',[25,240,590,180]); 
    axFreq = axes('Units','Pixels','Position',[25,25,590,180]); 

    % Assign the GUI a name to appear in the window title.
    set(f,'Name','Brain Vision RDA Client for MATLAB')
    % Move the GUI to the center of the screen.
    movegui(f,'center')
    % Make the GUI visible.
    set(f,'Visible','on'); 

    
%% ***********************************************************************   
% --- Closing reques handler: executes when user attempts to close formRDA.
function RDA_CloseRequestFcn(hObject, eventdata)
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB

    selection = questdlg(['Close MATLAB RDA Client?'],...
                         ['Closing...'],...
                         'Yes','No','Yes');
    if strcmp(selection,'No')
        return;
    end

    % Close open connections to recorder if exist
    CloseConnection();
    
    % Delete and close the window
    delete(hObject);


%% ***********************************************************************   
% Connection handling functions 

% --- Pushbutton handler: executes on button press in btConnect.
function btConnect_Callback(hObject, eventdata)
    % hObject    handle to btConnect
    % eventdata  reserved - to be defined in a future version of MATLAB

    text = get(hObject, 'String');
    if strcmp(text, 'Connect')
        OpenConnection();
        set(hObject,'String','Disconnect');
    else
        CloseConnection();
        set(hObject,'String','Connect');    
    end


% Connection opening
function OpenConnection()

    % global definitions
    global readTimer;       % Timer object for reading data from tcpip socket 
    global con;             % TCPIP connection
    global editHost;        % Host Name or IP edit control

    recorderip = get(editHost, 'String');

    % Establish connection to BrainVision Recorder Software 32Bit RDA-Port
    % (use 51234 to connect with 16Bit Port)
    con=pnet('tcpconnect', recorderip, 51244);

    % Check established connection and display a message
    stat=pnet(con,'status');
    if stat > 0
        disp('connection established');
    end

    % Define and start timer for reading from socket
    readTimer = timer('TimerFcn', @RDATimerCallback, 'Period', 0.01, 'ExecutionMode', 'fixedSpacing');
    start(readTimer);


% Connection closing
function CloseConnection()

    % global definitions
    global readTimer;       % Timer object

    % Stop the timer
    stop(readTimer);
    
    % Close all open socket connections
    pnet('closeall');
    
    % Display a message
    disp('connection closed');

    
    
%% ***********************************************************************   
% Callback function for RDA Timer
% 
% after a connection is established, this is the main data processing 
% function.
%
function RDATimerCallback(hObject, eventdata)
    % hObject    handle to timer object
    % eventdata  reserved - to be defined in a future version of MATLAB

    % global definitions
    global con;             % TCPIP connection
    global lastBlock;       % Number of last block for overflow test
    global props;           % EEG Properties
    global readTimer;       % Timer object
    global axTime;          % Time domain axes control
    global axFreq;          % Frequency domain axes control
    global hTime;           % Time domain graph handle
    global hFreq;           % Frequency domain graph handle
    global data1s;          % EEG data of the last recorded second

    % --- Main reading loop ---
    header_size = 24;
    try
        % check for existing data in socket buffer
        tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
        while ~isempty(tryheader)
            
            % Read header of RDA message
            hdr = ReadHeader(con);

            % Perform some action depending of the type of the data package
            switch hdr.type
                case 1       
                    %% Start, Setup information like EEG properties
                    disp('Start');
                    % Read and display EEG properties
                    props = ReadStartMessage(con, hdr);
                    disp(props);
                    
                    % Reset block counter to check overflows
                    lastBlock = -1;

                    % Fill data buffer with zeros and plot first time to
                    % get handles
                    data1s = zeros(props.channelCount,1000000 / props.samplingInterval);
                    hTime = plot(axTime, data1s(1,:));
                    freqEEGData = abs(fft(data1s(1,:)));
                    hFreq = plot(axFreq, freqEEGData(1:int32(length(freqEEGData)/2)));

                case 4       
                    %% 32Bit Data block
                    % Read data and markers from message
                    [datahdr, data, markers] = ReadDataMessage(con, hdr);
                    
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
                    end

                    % plot first channel using graph handle for better
                    % performance
                    set(axTime,'YLim',[-200, 200]);
                    set(hTime,'YData', data1s(1,:));
                    
                    % perform fourier transform of first channel
                    freqEEGData = abs(fft(data1s(1,:)));
                    
                    % plot fourier transform of first channel using graph
                    % handle for better performance
                    set(axFreq,'YLim',[0, 800]);
                    set(hFreq, 'YData', freqEEGData(1:int32(length(freqEEGData)/2)));

                case 3       
                    %% Stop message   
                    disp('Stop');
                    data = pnet(con, 'read', hdr.size - header_size);
 
                otherwise    % ignore all unknown types, but read the package from buffer 
                    data = pnet(con, 'read', hdr.size - header_size);
            end
            tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
        end
    catch
        er = lasterror;
        disp(er.message);
        % stop timer and close connection in case of an error
        stop(readTimer);
        pnet('closeall');
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
function [datahdr, data, markers] = ReadDataMessage(con, hdr)
    % con       tcpip connection object    
    % hdr       message header    
    % datahdr   data header with information on datalength and number of markers
    % data      data as one dimensional arry
    % markers   markers as array of marker structs
    
    % EEG properties as global
    global props;           % EEG Properties

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

