function varargout = BMI_Mahi_Closeloop_GUI(varargin)
% BMI_MAHI_CLOSELOOP_GUI MATLAB code for BMI_Mahi_Closeloop_GUI.fig
%      BMI_MAHI_CLOSELOOP_GUI, by itself, creates a new BMI_MAHI_CLOSELOOP_GUI or raises the existing
%      singleton*.
%
%      H = BMI_MAHI_CLOSELOOP_GUI returns the handle to a new BMI_MAHI_CLOSELOOP_GUI or the handle to
%      the existing singleton*.
%
%      BMI_MAHI_CLOSELOOP_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BMI_MAHI_CLOSELOOP_GUI.M with the given input arguments.
%
%      BMI_MAHI_CLOSELOOP_GUI('Property','Value',...) creates a new BMI_MAHI_CLOSELOOP_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BMI_Mahi_Closeloop_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BMI_Mahi_Closeloop_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BMI_Mahi_Closeloop_GUI

% Last Modified by GUIDE v2.5 24-Jul-2015 11:09:50
%% ********************Revisions
% 4-9-2015 - Stopped saving EMG values in .txt file 
% 7-24-2015 - Stopped launching flexion_game if use_mahi_exo was selected
% 8-10-2015 - Added and modified video recording function
%                     - Added second Serial COM port for triggering video capture in remote device
% 8-22-2015 - Replaced low pass and high pass filters with a single band pass filter [0.1 - 1 Hz]
% 8-24-2015 - Added new formula for computing trigger stamp when EEG_GO and EEG_EMG_GO occurs
% 9-08-2014 - Reverting back to high pass followed by low pass filter instead of band pass filter because of filter order and stability issues.
%                     - Changing order of low pass and large laplacian filters - should not affect output
% 9/14/2015 - Added baseline correction for EMG RMS calculation. The baseline is taken as 100 samples or ~ 30 sec during rest. 
% 9/25/2015 - Added 'S 80' marker to be saved in BMI file
%--------------------------------------------------------------------------------------------------
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BMI_Mahi_Closeloop_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @BMI_Mahi_Closeloop_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before BMI_Mahi_Closeloop_GUI is made visible.
function BMI_Mahi_Closeloop_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BMI_Mahi_Closeloop_GUI (see VARARGIN)

% Choose default command line output for BMI_Mahi_Closeloop_GUI
handles.output = hObject;

% Initialization code comes here - Nikunj
handles.user = varargin{1};
handles.user.testing_data.classifier_filename = '';
handles.user.testing_data.emg_decisions_fileID = 0;

set(handles.textbox_subject_initials,'String',handles.user.calibration_data.subject_initials);
set(handles.textbox_session_num,'String',num2str(handles.user.calibration_data.sess_num));
set(handles.textbox_block_num,'String',num2str(handles.user.calibration_data.block_num));
set(handles.textbox_cond_num,'String',num2str(handles.user.calibration_data.cond_num));

set(handles.textbox_cloop_session_num,'String',num2str(handles.user.testing_data.closeloop_sess_num));
set(handles.textbox_cloop_block_num,'String',num2str(handles.user.testing_data.closeloop_block_num));

set(handles.textbox_comm_port_num,'String','--','Enable','off');
set(handles.slider_counts_threshold,'value',1);
set(handles.slider_emg_threshold,'value',1);
set(handles.editbox_prob_threshold,'Enable','off');
set(handles.editbox_counts_threshold,'Enable','off');
set(handles.editbox_emg_threshold,'Enable','off');
set(handles.editbox_triceps_threshold,'Enable','off');
set(handles.pushbutton_initialize_conn,'Enable','off');

handles.system.serial_interface.comm_port_num = 0;
handles.system.serial_interface.enable_serial = 0;
handles.system.use_mahi_exo = 0;
handles.system.use_eeg = 0;
handles.system.use_eeg_emg = 0;
handles.system.use_emg = 0;
handles.system.left_impaired = 0;
handles.system.right_impaired = 0;


handles.system.serial_interface.serial_object = varargin{2};
if nargin > 2   % Serial port for webcam was detected
    handles.user.testing_data.serial_vidobj = varargin{3};
    % COM-port is already open. No need to call fopen().
else
    handles.user.testing_data.serial_vidobj = -1;
end

set(handles.axes_raster_plot,'box','on','nextplot','replacechildren');

handles.reset_eeg_control_timer = timer('TimerFcn', {@reset_eeg_control_timer_Callback,hObject},...
                                     'StartDelay', 0, 'Period',1,'ExecutionMode','fixedRate');

% Start flexion_game gui
% eval('flexion_game');                     % commented 7-24-2015
global hflexion_game
global datahdr marker_block

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BMI_Mahi_Closeloop_GUI wait for user response (see UIRESUME)
uiwait(handles.figure1);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)    
    
% % Hint: delete(hObject) closes the figure
% delete(hObject);
global hflexion_game

if ~isempty(hflexion_game)
    if ishandle(hflexion_game.figure_flexion_game) && strcmp(get(hflexion_game.figure_flexion_game, 'type'), 'figure')
        % findobj('type','figure','name','mytitle')
        close(hflexion_game.figure_flexion_game);
    end
end

if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end

% --- Outputs from this function are returned to the command line.
function varargout = BMI_Mahi_Closeloop_GUI_OutputFcn(hObject, eventdata, handles)  %#ok<*INUSL>
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure    
%varargout{1} = handles.output;

global processed_eeg processed_emg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold
global downsamp_factor emg_rms_baseline
%global unprocessed_eeg % test_change

varargout{1} = processed_eeg;       
varargout{2} = Overall_spatial_chan_avg;
varargout{3} = all_feature_vectors;
varargout{4} = GO_Probabilities;
varargout{5} = move_counts;
varargout{6} = marker_block;
varargout{7} = all_cloop_prob_threshold;
varargout{8} = all_cloop_cnts_threshold;
%varargout{9} = downsample(processed_emg',downsamp_factor)';
if ~isempty(processed_emg)
    %varargout{9} = resample(processed_emg',6,1)';    % Not required to resample - 9/14/2015
     %resample(cl_BMI_data$processed_emg[1,],20,3.333) - EMG RMS sampling frequency is 3.333 Hz
    varargout{9} = processed_emg;
else
    varargout{9} = 0;
end
varargout{10} = emg_rms_baseline; 

% The figure can be deleted now
delete(handles.figure1);

function textbox_subject_initials_Callback(hObject, eventdata, handles)
% hObject    handle to textbox_subject_initials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textbox_subject_initials as text
%        str2double(get(hObject,'String')) returns contents of textbox_subject_initials as a double

handles.user.calibration_data.subject_initials = get(hObject,'String');
guidata(hObject, handles);
        function textbox_subject_initials_CreateFcn(hObject, eventdata, handles) %#ok<*INUSD>
        % hObject    handle to textbox_subject_initials (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function textbox_session_num_Callback(hObject, eventdata, handles)
handles.user.calibration_data.sess_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_session_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_session_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function textbox_block_num_Callback(hObject, eventdata, handles)
handles.user.calibration_data.block_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_block_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_block_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function textbox_cond_num_Callback(hObject, eventdata, handles)
handles.user.calibration_data.cond_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_cond_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_cond_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function textbox_cloop_session_num_Callback(hObject, eventdata, handles)
handles.user.testing_data.closeloop_sess_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_cloop_session_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_cloop_session_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function textbox_cloop_block_num_Callback(hObject, eventdata, handles)
handles.user.testing_data.closeloop_block_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_cloop_block_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_cloop_block_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

%---------------------------------------------------------------------------------------------------
% --- Executes when figure1 is resized.
function figure1_ResizeFcn(hObject, eventdata, handles)

function textbox_comm_port_num_Callback(hObject, eventdata, handles)
handles.system.serial_interface.comm_port_num = str2double(get(hObject,'String'));
guidata(hObject, handles);
        function textbox_comm_port_num_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to textbox_comm_port_num (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end

function checkbox_enable_serial_Callback(hObject, eventdata, handles)
handles.system.serial_interface.enable_serial = get(hObject,'Value');
% % persistent serial_obj
% % if handles.system.serial_interface.enable_serial
% %     serial_obj = serial(['com' num2str(handles.system.serial_interface.comm_port_num)],...
% %         'baudrate',9600,'parity','none','databits',8,'stopbits',1); %Mahi
% %     fopen(serial_obj);
% %     handles.system.serial_interface.serial_object = serial_obj;
% % end
% % 
% % if ~isempty(handles.system.serial_interface.serial_object) && handles.system.serial_interface.enable_serial == 0
% %     fclose(serial_obj);
% %     handles.system.serial_interface.serial_object = [];
% % end
guidata(hObject, handles);    

function pushbutton_initialize_conn_Callback(hObject, eventdata, handles)
    
function pushbutton_start_closeloop_Callback(hObject, eventdata, handles)
    
    global readEEGTimer     %#ok<*NUSED> % Timer object for reading data from tcpip socket
    global con              % TCPIP connection
    global classifier_channels
    global emg_classifier_channels
    global resamp_Fs
    global window_length
    global window_time
    global CloopClassifier
    global Best_BMI_classifier
    global cloop_prob_threshold
    global cloop_cnts_threshold
    global emg_mvc_threshold
    global emg_tricep_threshold
    global DEFINE_GO
    global DEFINE_NOGO
    global target_trig_label
    global move_trig_label
    global rest_trig_label
    global block_start_label block_stop_label
    global eeg_control
    global emg_control_channels
    global hflexion_game
    
    % Load Performance variable from .mat file
    handles.user.calibration_data.folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' handles.user.calibration_data.subject_initials '\' ...
                                                handles.user.calibration_data.subject_initials '_Session'...
                                                num2str(handles.user.calibration_data.sess_num) '\'];   %change15
                                            
    handles.user.testing_data.closeloop_folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' handles.user.calibration_data.subject_initials '\' ...
                                                handles.user.calibration_data.subject_initials '_Session'...
                                                num2str(handles.user.testing_data.closeloop_sess_num) '\'];     %change16
    
                                            
    handles.user.testing_data.classifier_filename = [handles.user.calibration_data.folder_path...
                 handles.user.calibration_data.subject_initials...
                 '_ses' num2str(handles.user.calibration_data.sess_num)...
                 '_cond' num2str(handles.user.calibration_data.cond_num)...
                 '_block' num2str(handles.user.calibration_data.block_num)...
                 '_performance_optimized_conventional_smart.mat'];           % change17
   
   handles.user.testing_data.closeloop_block_num = handles.user.testing_data.closeloop_block_num + 1;
   set(handles.textbox_cloop_block_num,'String',num2str(handles.user.testing_data.closeloop_block_num));          
   
   % commented 4-9-15
   %handles.user.testing_data.emg_decisions_fileID = fopen([handles.user.testing_data.closeloop_folder_path handles.user.calibration_data.subject_initials...
   %            '_ses' num2str(handles.user.testing_data.closeloop_sess_num) '_block' num2str(handles.user.testing_data.closeloop_block_num)...
   %            '_closeloop_emg_decisions.txt'],'w');
   
   % commented 4-9-15
   %fprintf(handles.user.testing_data.emg_decisions_fileID,'rms_biceps \t rms_triceps \t bicep_thr \t tricep_thr \t Decision \t Timer_status \n');            
            
    load(handles.user.testing_data.classifier_filename);
    CloopClassifier = Performance;
    % classifier_channels = CloopClassifier.classchannels;
    classifier_channels =   Performance.optimized_channels; % change18 - Added 9/7/2015
    disp(classifier_channels);
    emg_classifier_channels = [42 41 51 17 45 46 55 22]; % [LB1 LB2 LT1 LT2 RB1 RB2 RT1 RT2] % change19
        
    if handles.system.left_impaired 
        emg_control_channels = [1 2];
    elseif handles.system.right_impaired
        emg_control_channels = [3 4];
    else 
        errordlg('Please select which hand is impaired','EMG control error');
        return
    end
    set(handles.checkbox_left_hand_impaired,'Enable','off');
    set(handles.checkbox_right_hand_impaired,'Enable','off');

    resamp_Fs = 20; % required Sampling Freq
    window_length = CloopClassifier.smart_window_length*resamp_Fs + 1; % No. of samples required, Earlier  = length(CloopClassifier.move_window(1):1/resamp_Fs:CloopClassifier.move_window(2));   
    window_time = 1/resamp_Fs:1/resamp_Fs:window_length/resamp_Fs;

    % Decide classifier with best accuracy
    [max_acc_val,max_acc_index] = max(CloopClassifier.eeg_accur); 
    Best_BMI_classifier = CloopClassifier.eeg_svm_model{max_acc_index};

    % Classifier parameters
    cloop_prob_threshold = 0.643; %CloopClassifier.opt_prob_threshold;            % change20 
    cloop_cnts_threshold  = 5; %CloopClassifier.consecutive_cnts_threshold;        % Number of consecutive valid cnts required to accept as 'GO'
    emg_mvc_threshold = 2;             % change21       % Biceps
    emg_tricep_threshold = 2;          % Triceps

    DEFINE_GO = 1;
    DEFINE_NOGO = 2;
    target_trig_label = 'S  8';                 
    move_trig_label = 'S 16';  % 'S 32'; %'S  8'; %'100';   
    rest_trig_label = 'S  2';  % 'S  2'; %'200';
    block_start_label = 'S 42'; %'S 10';        % change22
    block_stop_label = 'S238';
    
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
    
    % Define and start timer for reading from socket every 10 msec
    readEEGTimer = timer('TimerFcn', {@EEGTimer_Callback,hObject}, 'Period', 0.01, 'ExecutionMode','fixedRate'); 
    start(readEEGTimer);
    eeg_control =  0;
    
    
    set(handles.slider_prob_threshold,'value',cloop_prob_threshold);
    set(handles.editbox_prob_threshold,'String',num2str(cloop_prob_threshold));
    set(handles.slider_counts_threshold,'value',cloop_cnts_threshold);
    set(handles.editbox_counts_threshold,'String',num2str(cloop_cnts_threshold));
    set(handles.slider_emg_threshold,'value',emg_mvc_threshold);
    set(handles.slider_tricep_threshold,'value',emg_tricep_threshold);
    set(handles.editbox_emg_threshold,'String',num2str(emg_mvc_threshold));
    set(handles.editbox_triceps_threshold,'String',num2str(emg_tricep_threshold));
    
    if ~handles.system.use_mahi_exo
        eval('flexion_game');                     % added 7-24-2015
        %disp(hflexion_game);
    else
        %disp(hflexion_game);
    end
    
    % Adding video capture  - 8/10/2015
    % [handles.user.testing_data.vidobj,handles.user.testing_data.capture_fig] = start_video_capture(handles);  % commented 8-11-2015
    if handles.user.testing_data.serial_vidobj ~= -1
        video_filename = [handles.user.calibration_data.subject_initials '_ses' num2str(handles.user.testing_data.closeloop_sess_num)...
                                          '_block' num2str(handles.user.testing_data.closeloop_block_num) '_closeloop_video_' datestr(clock,'mm-dd-yyyy_HH_MM_SS')];
        fprintf(handles.user.testing_data.serial_vidobj,'%s\n',['Start ' video_filename]);       % send start message and filename to remote webcam, SPACE after Start is required
    end
    
    guidata(hObject,handles);

function pushbutton_stop_closeloop_Callback(hObject, eventdata, handles)
% global definitions
    global processed_eeg processed_emg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
    global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold datahdr
    global downsamp_factor
    
    global readEEGTimer;       % Timer object   
    % Stop the timer
    stop(readEEGTimer);
    % Close all open socket connections
    pnet('closeall');
    % Display a message
    disp('connection closed');
    
    if get(handles.checkbox_auto_save_data,'Value')
        save_cloop_data(handles);
    end
       % commented 4-9-15
       %fclose(handles.user.testing_data.emg_decisions_fileID);
       
     % Adding video capture  
    if handles.user.testing_data.serial_vidobj ~= -1
        % stop_video_capture(handles); % commented 8-11-2015
        fprintf(handles.user.testing_data.serial_vidobj,'%s\n','Stop ');       % send stop message to remote webcam, SPACE after Stop is required
    else
        warndlg('Video was not recorded');
    end
    % close(handles.user.testing_data.capture_fig);     % commented 8-11-2015
          
       
    %% Raster plot for close loop 
try       
                    Proc_EEG = processed_eeg; 
                    %Proc_EMG = downsample(processed_emg',downsamp_factor)';
                    Proc_EMG = resample(processed_emg',6,1)';
                    Ovr_Spatial_Avg = Overall_spatial_chan_avg;
                    All_Feature_Vec = all_feature_vectors;
                    GO_Prob = GO_Probabilities;
                    Num_Move_Counts = move_counts;
                    Markers = marker_block;

                    raw_Fs = 500;
                    proc_Fs = 20;

                    event_times = (double(Markers(:,1))/raw_Fs);    
                    pred_start_stop_times = round(event_times(Markers(:,2)==50).*proc_Fs)/proc_Fs;
                    if length(pred_start_stop_times) == 1
                        pred_start_stop_times = [pred_start_stop_times(1) round(event_times(end).*proc_Fs)/proc_Fs];
                    end

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
                    raster_zscore = raster_data;
                    raster_time = pred_eeg_time;
                    raster_colors = ['m','k','b','r','k','b','r'];
                    % Plot the rasters; Adjust parameters for plot
                    [raster_row,raster_col] = size(raster_zscore);
                    add_offset = 5;
                    raster_zscore(:,1:3) = raster_zscore(:,1:3).*0.1;
                    raster_zscore(:,5) = raster_zscore(:,5).*0.1;
                    raster_zscore(:,4) = raster_zscore(:,4).*0.1;
                    
                    for raster_index = 1:raster_col;
                        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index+20;  % Add offset to each channel of raster plot
                        myhand(raster_index) = plot(handles.axes_raster_plot,raster_time,raster_zscore(:,raster_index),raster_colors(raster_index));
                        hold on;
                    end
                    set(myhand(5),'LineWidth',2);
                    set(myhand(6),'LineWidth',2);
                    set(myhand(7),'LineWidth',2);
                    %plot(raster_time, 10/3*Num_Move_Counts, '--b','LineWidth',2);
                    hold on;
                    plot(handles.axes_raster_plot,raster_time, 10 + 10*GO_Prob, '.r','LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, 10 + 10*all_cloop_prob_threshold(1:length(pred_Ovr_Spatial_Avg)),'-','Color',[0.4 0.4 0.4],'LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, all_cloop_cnts_threshold(1:length(pred_Ovr_Spatial_Avg)),'-k','LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, Num_Move_Counts,'b');
                    %line([0 200],10.*[cloop_prob_threshold cloop_prob_threshold],'Color',[0.4 0.4 0.4],'LineWidth',2);
                    axis(handles.axes_raster_plot,[pred_start_stop_times(1) pred_start_stop_times(2) 1 70]);
                    %ylim([0 40]);

                    myaxis = axis;
                    for plot_ind1 = 1:length(pred_stimulus_times);
                        line([pred_stimulus_times(plot_ind1), pred_stimulus_times(plot_ind1)],[myaxis(3)-1, myaxis(4)],'Color','g','LineWidth',1);
                        text(pred_stimulus_times(plot_ind1)-1,myaxis(4)+0.5,'Start','Rotation',60,'FontSize',12);
                        %text(pred_stimulus_times(plot_ind1)-1.5,myaxis(4)+1,'shown','Rotation',0,'FontSize',12);
                        hold on;
                    end

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

%                     for plot_ind3 = 1:length(pred_move_onset_times);
%                         line([pred_move_onset_times(plot_ind3), pred_move_onset_times(plot_ind3)],[myaxis(3), myaxis(4)],'Color','r','LineWidth',2,'LineStyle','--');
%                         hold on;
%                     end
                    set(handles.axes_raster_plot,'YTick',[1 all_cloop_cnts_threshold(1) 5 10 10+10*all_cloop_prob_threshold(1) 20 25 30 35 40 45 50 55]);
                    set(handles.axes_raster_plot,'YTickLabel',{'1' 'Count_Thr' '5' 'p(GO) = 0','Prob_Thr','p(GO) = 1', 'AUC', '-ve Peak','Slope','Mahal','Spatial Avg','Biceps','Triceps'});
                    xlabel(handles.axes_raster_plot,'Time (sec.)', 'FontSize' ,12);
                    %title(handles.axes_raster_plot,'Raster Plot','FontSize',12);
                    %export_fig 'Block8_results' '-png' '-transparent'
                    hold off;    
catch raster_err
        cla(handles.axes_raster_plot,'reset')    
        axes_xlim = get(handles.axes_raster_plot,'XLim');
        axes_ylim = get(handles.axes_raster_plot,'YLim');
        axes(handles.axes_raster_plot);
        text(axes_xlim(1) + 0.2, axes_ylim(2)/2,'Error: Insufficient data to create plot','FontSize',12,'Color','r');
        text(axes_xlim(1) + 0.2, axes_ylim(2)/2-0.2, raster_err.message,'Color','r');
end
    
    guidata(hObject,handles);

%% ***********************************************************************   
% Callback function for EEGTimer
% 
% after a connection is established, this is the main data processing 
% function.
%
function EEGTimer_Callback(timer_object,eventdata,hObject)
    
    handles = guidata(hObject);
    % global definitions
    % 1. Variables initialized in pushbutton_start_closeloop_Callback()
    global con;             % TCPIP connection
    global readEEGTimer;       % Timer object
    global classifier_channels
    global emg_classifier_channels
    global resamp_Fs
    global window_length
    global window_time
    global CloopClassifier
    global Best_BMI_classifier
    global cloop_prob_threshold
    global cloop_cnts_threshold
     global emg_mvc_threshold
     global emg_tricep_threshold
    global DEFINE_GO
    global DEFINE_NOGO
    global target_trig_label
    global move_trig_label
    global rest_trig_label
    global block_start_label block_stop_label
    global hflexion_game
    global datahdr
    global eeg_control
    global emg_control_channels
    
    % 2. Variables initialized in EEGTimer_Callback()
    global orig_Fs
    global lastBlock;       % Number of last block for overflow test
    global props;           % EEG Properties
    global dataX00ms dataEMGms         % EEG data of the last recorded second
    global total_chnns
    global downsamp_factor pkt_size emg_pkt_size 
    global new_pkt eeg_bp_filt_pkt spatial_filt_pkt emg_bp_filt_pkt
    global eeg_bpf_df2sos emg_bpf_df2sos prev_eeg_bp_filt_pkt_3D prev_emg_bp_filt_pkt_3D
    global emg_new_pkt emg_diff_pkt emg_rms_pkt emg_rms_baseline update_emg_baseline
    global firstblock marker_block marker_type
    global processed_eeg processed_emg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
    global valid_move_cnt move_counts feature_matrix sliding_window start_prediction kcount
    global all_cloop_prob_threshold all_cloop_cnts_threshold
    global required_chnns L_LAP M_CAR
    global filt_order hp_filt_pkt lp_filt_pkt hpf_df2sos lpf_df2sos prev_lp_filt_pkt_3D prev_hp_filt_pkt_3D % Parameters for low pass and high pass EEG filters 
    
    % ----- unused variables - 8/22/2015
    % global unprocessed_eeg 
    
    %global prev_pkt prev_spatial_filt_pkt
    %global  prev_hp_filt_pkt  prev_lp_filt_pkt prev_emg_bp_filt_pkt
    %global emg_prev_pkt prev_emg_abs_diff_pkt prev_emg_mvc_pkt prev_emg_diff_pkt emg_abs_diff_pkt emg_mvc_pkt
    
            %% ------------------ Main reading loop ----------------------------------------
        header_size = 24;
        %finish = false;
        %while ~finish
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
                            disp('Start data streaming...');
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
                            total_chnns = double(props.channelCount);
                            new_pkt = zeros(total_chnns,pkt_size);
                            %prev_pkt = zeros(total_chnns,pkt_size);
                            hp_filt_pkt = zeros(total_chnns,pkt_size);
                            %prev_hp_filt_pkt =  zeros(total_chnns,pkt_size);
                            prev_hp_filt_pkt_3D = zeros(2,2,total_chnns);                                                                                                              
                            % lp_filt_pkt = zeros(length(classifier_channels),pkt_size);      % Use only classifier channels; commented on 9/8/2015
                            lp_filt_pkt = zeros(total_chnns,pkt_size);
                            %prev_lp_filt_pkt = zeros(length(classifier_channels),pkt_size);     
                            % prev_lp_filt_pkt_3D = zeros(2,2,length(classifier_channels)); % commented on 9/8/2015
                            prev_lp_filt_pkt_3D = zeros(2,2,total_chnns); 
                            eeg_bp_filt_pkt = zeros(total_chnns,pkt_size);
                            prev_eeg_bp_filt_pkt_3D = zeros(2,2,total_chnns);     
                            spatial_filt_pkt = zeros(total_chnns,pkt_size);
                            %prev_spatial_filt_pkt = zeros(total_chnns,pkt_size);
                            
                            % Create 0.1 Hz high pass filter dfilt object
                            [b_hpf,a_hpf] = butter(4,(0.1/(orig_Fs/2)),'high');
                            SOS_hpf = tf2sos(b_hpf,a_hpf);
                            hpf_df2sos = dfilt.df2sos(SOS_hpf);
                            hpf_df2sos.PersistentMemory = true;
                            
                            % Create 1 Hz low pass filter dfilt object
                            [b_lpf,a_lpf] = butter(4,(1/(orig_Fs/2)),'low');
                            SOS_lpf = tf2sos(b_lpf,a_lpf);
                            lpf_df2sos = dfilt.df2sos(SOS_lpf);
                            lpf_df2sos.PersistentMemory = true;
                            
                            % Create EEG 0.1 - 1 Hz band pass filter dfilt object 
                            [eeg_b_bpf,eeg_a_bpf] = butter(2,([0.1 1]./(orig_Fs/2)),'bandpass');
                            eeg_SOS_bpf = tf2sos(eeg_b_bpf,eeg_a_bpf);
                            eeg_bpf_df2sos = dfilt.df2sos(eeg_SOS_bpf);
                            eeg_bpf_df2sos.PersistentMemory = true;
                            
                            % Create EMG band-pass filter (30 - 200 Hz) dfilt object
                            [emg_b_bpf,emg_a_bpf] = butter(2,([30 200]./(orig_Fs/2)),'bandpass');
                            emg_SOS_bpf = tf2sos(emg_b_bpf,emg_a_bpf);
                            emg_bpf_df2sos = dfilt.df2sos(emg_SOS_bpf);
                            emg_bpf_df2sos.PersistentMemory = true;

                            firstblock = true;          % First block of data read in 
                            marker_block = [];
                            marker_type = [];

                            processed_eeg = [];
                            processed_emg = [];
                            % unprocessed_eeg = [];
                            Overall_spatial_chan_avg = [];
                            %lpf_eeg = [];
                            %raster_signal = [];
                            all_feature_vectors = [];
                            all_cloop_prob_threshold = [];
                            all_cloop_cnts_threshold = [];
                            GO_Probabilities = [];
                            valid_move_cnt = 0;
                            move_counts = [];

                            feature_matrix = zeros(length(classifier_channels),pkt_size/downsamp_factor);
                            sliding_window = zeros(1,window_length); %zeros(length(classifier_channels),window_length);
                            %move_command = [];           % Number of predicted moves
                            %rest_command = [];           % Number of predicted rests
                            %likert_score = [];          % To be used for recording trial success/ failure
                            %prediction = [];
                            if handles.system.use_mahi_exo == 1
                                   start_prediction = false;            
                            else
                                   start_prediction = true;
                            end
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

                            % Create Large Laplacian Matrix - Using larger Laplacian filter
                            if total_chnns == 64
                            Neighbors = [
                                         % Row F
                                         4,  1, 3, 13, 5;
                                        38, 1, 37, 48, 39;
                                         5, 28, 4, 14, 6;       
                                        39, 2, 38, 49, 40;
                                         6,  2, 5, 15, 7;
                                         % Row FC
                                         43, 34, 8, 52, 32;        % Instead of Ch. 42 using  Ch. 8, because Ch. 42 is now being used for recording EMG - 8/28/2015
                                          9, 34, 8, 19, 10;
                                         32, 28, 43, 53, 44;    
                                         10, 35, 9, 11, 20;
                                         44, 35, 32, 54, 11;    % Instead of Ch. 45 using  Ch. 11
                                         % Row C
                                         13, 4, 12, 24, 14;
                                         48, 38, 47, 57, 49;
                                         14, 5, 13, 25, 15;
                                         49, 39, 48, 58, 50;
                                         15, 6, 14, 26, 16;
                                         % Row CP
                                          52, 43, 18, 60, 53;    % Instead of Ch. 51 using  Ch. 18
                                         19, 9, 18, 61, 20; 
                                         53, 32, 52, 62, 54;
                                         20, 10, 19, 63, 21;
                                         54, 44, 53, 64, 21;    % Instead of Ch. 55 using  Ch. 21
                                         % Row P
                                         24, 13, 23, 29, 25;
                                         57, 48, 56, 30, 58;
                                         25, 14, 24, 30, 26;
                                         58, 49, 57, 31, 59;
                                         26, 15, 25, 31, 27];
                            elseif total_chnns == 32
                            % For modified 32 channels          
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
                            [datahdr, data, markers] = ReadDataMessage(con, hdr);   % Removed props; Nikunj - 7/9/14
                            if firstblock == true
                                marker_block = [marker_block; [datahdr.block,1]];
                                
                                % Initialize EMG variables here because
                                % packet size changes for real-time and simulated EEG recordings 
                                % emg_pkt_size = datahdr.points;
                                % EMG sampling rate works out to be 3.33 Hz
                                emg_pkt_size = int32((6/resamp_Fs)*orig_Fs);           % 6 samples --> 300ms @ 20 Hz, 150 samples --> 300ms @ 500 Hz; EMG window length = 300 ms
                                emg_new_pkt = zeros(length(emg_classifier_channels),emg_pkt_size);
                                %emg_prev_pkt = zeros(length(emg_classifier_channels),emg_pkt_size);
                                emg_bp_filt_pkt = zeros(length(emg_classifier_channels),emg_pkt_size);      % Use only EMG channels
                                %prev_emg_bp_filt_pkt = zeros(length(emg_classifier_channels),emg_pkt_size);
                                prev_emg_bp_filt_pkt_3D = zeros(2,2,length(emg_classifier_channels));
                                % emg_abs_diff_pkt = zeros(2,emg_pkt_size);
                                emg_diff_pkt = zeros(4,emg_pkt_size);
                                % emg_mvc_pkt = zeros(2,emg_pkt_size);
                                emg_rms_pkt = [0;0;0;0];
                                emg_rms_baseline = [0;0;0;0];
                                update_emg_baseline = 0;
                                % prev_emg_abs_diff_pkt = zeros(2,emg_pkt_size);
                                % prev_emg_mvc_pkt = zeros(2,emg_pkt_size);
                                % prev_emg_diff_pkt = zeros(2,emg_pkt_size);
                                
                                
                                if start_prediction == true     % Used when calling ball-game GUI                                                                          
                                    %marker_block = [marker_block; [datahdr.block*datahdr.points,50]];      % Not completely accurate
                                    marker_block = [marker_block; [((datahdr.block - 1)*datahdr.points + 1),50]];  % Added 8/24/2015
                                end
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
                                    if (strcmp(markers(m).description,move_trig_label) || strcmp(markers(m).description,'S 80'))         % Movement onset                                        
                                        marker_block = [marker_block; [double(markers(m).sample_num),100]];
                                        %kcount = kcount + 1;
                                    elseif (strcmp(markers(m).description,rest_trig_label))     % Targets appear
                                       % baseline = feature_data;  
                                       % new_baseline = true;
                                       marker_block = [marker_block; [double(markers(m).sample_num),2]];
                                   elseif (strcmp(markers(m).description, target_trig_label))           % Target reached
                                        marker_block = [marker_block; [double(markers(m).sample_num),8]];
                                    elseif (strcmp(markers(m).description, 'S  4'))                      % next is catch trials
                                        marker_block = [marker_block; [double(markers(m).sample_num),4]];
                                    elseif (strcmp(markers(m).description,'S 12'))                      % Target reached + next is catch trial
                                        marker_block = [marker_block; [double(markers(m).sample_num),8]; [double(markers(m).sample_num),4]];
                                    elseif (strcmp(markers(m).description,'S  1'))                      % Timeout occured
                                        marker_block = [marker_block; [double(markers(m).sample_num),350]];
                                    elseif (strcmp(markers(m).description,'S  64'))                      % EEG_GO prediction occured
                                        marker_block = [marker_block; [double(markers(m).sample_num),64]];
                                    elseif ((strcmp(markers(m).description,block_start_label))  || (strcmp(markers(m).description,block_stop_label)))  % Start/Stop Prediction
                                       start_prediction = ~start_prediction; 
                                       marker_block = [marker_block; [double(markers(m).sample_num),50]];
                                       if start_prediction == true
                                           disp('Starting Prediction');
                                           update_emg_baseline = 1;        % Added 9/14/2015
                                       else
                                           disp('Stoping Prediction');
                                       end
                                    end                                   
                                end                          
                            end

                            % Process raw EEG data, q
                            EEGData = reshape(data, props.channelCount, length(data) / props.channelCount);
                            EEGData = double(EEGData);  % Very Important!! Always convert to double precision
                            %EEGData = EEGData.*props.resolutions(1,1);
                            EEGData = diag(props.resolutions)*EEGData;
                            
%--------------------------------------Add EMG filters here. 
                               dataEMGms = [dataEMGms EEGData(emg_classifier_channels,:)];
                               if size(dataEMGms,2) > emg_pkt_size
                                        emg_new_pkt = dataEMGms(:,1:emg_pkt_size);
                                        dataEMGms = dataEMGms(:,emg_pkt_size+1 : size(dataEMGms,2));          % Move all excess data to next data packet       
                                       % EMG Bandpass Filter
%                                       emg_new_pkt = EEGData(emg_classifier_channels,:);
                                       for num_chns = 1:length(emg_classifier_channels)
                                            %emg_bp_filt_pkt(num_chns,:) = pkt_bp_filter(emg_new_pkt(num_chns,:),...
                                            % emg_prev_pkt(num_chns,(emg_pkt_size - filt_order)+1:emg_pkt_size),...
                                            % prev_emg_bp_filt_pkt(num_chns,(emg_pkt_size - filt_order)+1:emg_pkt_size),...
                                            % orig_Fs);
                                            
                                            emg_bpf_df2sos.States = prev_emg_bp_filt_pkt_3D(:,:,num_chns);
                                            emg_bp_filt_pkt(num_chns,:) = filter(emg_bpf_df2sos, emg_new_pkt(num_chns,:));
                                            prev_emg_bp_filt_pkt_3D(:,:,num_chns) = emg_bpf_df2sos.States;
                                       end

%                                        emg_abs_diff_pkt = [abs(emg_bp_filt_pkt(1,:) - emg_bp_filt_pkt(3,:));            % Biceps
%                                                                    abs(emg_bp_filt_pkt(2,:) - emg_bp_filt_pkt(4,:))];                  % Triceps
                                       % LPF filter to get EMG Envelop   
                                       % cutoff frequency to 2.2 Hz
%                                        for num_chns = 1:2
%                                            emg_mvc_pkt(num_chns,:) = pkt_lp_filter(emg_abs_diff_pkt(num_chns,:),...
%                                                prev_emg_abs_diff_pkt(num_chns,(emg_pkt_size - filt_order)+1:emg_pkt_size),...
%                                                prev_emg_mvc_pkt(num_chns,(emg_pkt_size - filt_order)+1:emg_pkt_size),...
%                                                orig_Fs);
%                                        end

                                           emg_diff_pkt = [emg_bp_filt_pkt(1,:) - emg_bp_filt_pkt(2,:);            % Left Biceps % changed 7-24-2015 
                                                                         emg_bp_filt_pkt(3,:) - emg_bp_filt_pkt(4,:);            % Left Triceps
                                                                         emg_bp_filt_pkt(5,:) - emg_bp_filt_pkt(6,:);            % Right Biceps
                                                                         emg_bp_filt_pkt(7,:) - emg_bp_filt_pkt(8,:)];           % Right Triceps   
                                        
                                       % Calcuate emg baseline for RMS - Added 9/14/2015
                                       if update_emg_baseline == 1
                                           % look back at 33 samples ~ 10 seconds; 100 samples ~ 30 sec
                                           emg_rms_baseline = mean(processed_emg(:,end-100+1:end),2);
                                           disp(emg_rms_baseline')
                                           update_emg_baseline = 0;
                                       end
                                       
                                       % Calculate EMG RMS value
                                           %emg_rms_pkt = [sqrt(mean(emg_diff_pkt(1,:).^2));...
                                            %                                                          sqrt(mean(emg_diff_pkt(2,:).^2))]; % commented 7-24-2015 
                                            
                                            % emg_rms_pkt = sqrt(mean(emg_diff_pkt.^2,2));  % commented on 9/14/2015
                                        emg_rms_pkt = sqrt(mean(emg_diff_pkt.^2,2)) - emg_rms_baseline; 
                                        
                                        % emg_prev_pkt = emg_new_pkt;
                                        % prev_emg_bp_filt_pkt = emg_bp_filt_pkt;
        %                                 prev_emg_abs_diff_pkt = emg_abs_diff_pkt;
        %                                 prev_emg_mvc_pkt = emg_mvc_pkt;
        %                                 processed_emg = [processed_emg emg_mvc_pkt];                            
                                        % prev_emg_diff_pkt = emg_diff_pkt;
                                        processed_emg = [processed_emg emg_rms_pkt];                        % Interpolate later on     
                               end
                                                             
%---------------------------------------------------------------------
                            dataX00ms = [dataX00ms EEGData];
                            dims = size(dataX00ms);                  
                            % For 700 msec windows, % For 500 msec windows: if dims(2) > (1000000 / props.samplingInterval)/2 
                            if dims(2) > pkt_size     
                               new_pkt = dataX00ms(:,1:pkt_size);                      % Select 64xN size of data packet i.e 2*N msec
                               dataX00ms = dataX00ms(:,pkt_size+1 : dims(2));          % Move all excess data to next data packet        
                              
                               % High pass filter 0.1 Hz - commented 8/22/2015; uncommented on 9/8/2015
                               for no_chns = 1:length(required_chnns)
%                                    hp_filt_pkt(required_chnns(no_chns),:) = pkt_hp_filter(new_pkt(required_chnns(no_chns),:),...
%                                        prev_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size),...
%                                        prev_hp_filt_pkt(required_chnns(no_chns),(pkt_size - filt_order)+1:pkt_size),...
%                                        orig_Fs);
                                   
                                   % High pass filter using dfilt()
                                   hpf_df2sos.States = prev_hp_filt_pkt_3D(:,:,required_chnns(no_chns));
                                   hp_filt_pkt(required_chnns(no_chns),:) = filter(hpf_df2sos,new_pkt(required_chnns(no_chns),:));
                                   prev_hp_filt_pkt_3D(:,:,required_chnns(no_chns)) = hpf_df2sos.States;
                               end                            
                               
%                                % Band pass filter 0.1-1 Hz, 8/22/2015; commented on 9/8/2015
%                                for no_chns = 1:length(required_chnns)                                 
%                                    eeg_bpf_df2sos.States = prev_eeg_bp_filt_pkt_3D(:,:,required_chnns(no_chns));
%                                    eeg_bp_filt_pkt(required_chnns(no_chns),:) = filter(eeg_bpf_df2sos,new_pkt(required_chnns(no_chns),:));
%                                    prev_eeg_bp_filt_pkt_3D(:,:,required_chnns(no_chns)) = eeg_bpf_df2sos.States;
%                                end             

                                    % Low pass filter 1 Hz, prev_spatial_filt_pkt; commented 8/22/2015; uncommented on 9/8/2015
                               for no_chns = 1:length(required_chnns)                                
%                                    lp_filt_pkt(no_chns,:) = pkt_lp_filter(spatial_filt_pkt(classifier_channels(no_chns),:),...
%                                        prev_spatial_filt_pkt(classifier_channels(no_chns),(pkt_size - filt_order)+1:pkt_size),...
%                                        prev_lp_filt_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size),...
%                                        orig_Fs);
                                        % Low pass filter using dfilt()
                                        lpf_df2sos.States = prev_lp_filt_pkt_3D(:,:,required_chnns(no_chns));
                                        %lp_filt_pkt(required_chnns(no_chns),:) = filter(lpf_df2sos,spatial_filt_pkt(required_chnns(no_chns),:)); % commented 9/8/2015
                                        lp_filt_pkt(required_chnns(no_chns),:) = filter(lpf_df2sos,hp_filt_pkt(required_chnns(no_chns),:));
                                        prev_lp_filt_pkt_3D(:,:,required_chnns(no_chns)) = lpf_df2sos.States;
                               end
                               
                               % Re-reference the data                                                     
                               %spatial_filt_pkt = M_CAR*hp_filt_pkt;      % Common Average referencing  
                               spatial_filt_pkt = L_LAP*lp_filt_pkt;        % Large Laplacian filtering; uncommented on 9/8/2015
                               %spatial_filt_pkt = L_LAP*eeg_bp_filt_pkt;   % commented 9/8/2015
                               % spatial_filt_pkt = L_LAP*hp_filt_pkt; % commented 9/8/2015
                                                              
                               

                               % Downsample data to 20 Hz
                               %Filtered_EEG_Downsamp = downsample(lp_filt_pkt',downsamp_factor)' - repmat(mean(baseline,2),1,window_size);
                               % Filtered_EEG_Downsamp = downsample(lp_filt_pkt',downsamp_factor)';        % No baseline correction - commented 8/22/2015
                               Filtered_EEG_Downsamp = downsample(spatial_filt_pkt(classifier_channels,:)',downsamp_factor)';
                               
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
                                 all_cloop_prob_threshold = [all_cloop_prob_threshold cloop_prob_threshold];
                                 all_cloop_cnts_threshold = [all_cloop_cnts_threshold cloop_cnts_threshold];
                                 
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
                                            if handles.system.use_eeg == 1      % If using EEG control only, then do not check EMG
                                                            if handles.system.serial_interface.enable_serial
                                                                fwrite(handles.system.serial_interface.serial_object,[7]);                
                                                            elseif strcmp(get(hflexion_game.hball,'Visible'),'on') && strcmp(get(hflexion_game.move_ball_timer,'Running'),'off')
                                                                start(hflexion_game.move_ball_timer);
                                                            else
                                                                %ignore
                                                            end                                                  
                                                            disp('--->EEG GO');
                                                            %marker_block = [marker_block; [datahdr.block*datahdr.points,300]];      % Not completely accurate
                                                            marker_block = [marker_block; [((datahdr.block - 1)*datahdr.points + 1),300]];      % Added 8/24/2015
                                            end

                                            if handles.system.use_eeg_emg == 1 % If using both EEG and EMG control, then check if EMG activity is present
                                                eeg_control = 1;
                                                fwrite(handles.system.serial_interface.serial_object,[5]);                              % To generate trigger pulse from Exoskeleton to EEG system == 'S 64'   
                                                % Start timer for resetting eeg_control, if
                                                % it is already not running ~ Important
                                                if strcmp(get(handles.reset_eeg_control_timer,'Running'),'off')
                                                    start(handles.reset_eeg_control_timer);
                                                end
                                                disp('--->EEG GO');
                                                %marker_block = [marker_block; [datahdr.block*datahdr.points,300]];      % Not completely accurate
                                                marker_block = [marker_block; [((datahdr.block - 1)*datahdr.points + 1),300]];      % Added 8/24/2015
                                            end

                                       end % endif
                                       %move_command = [move_command; 1];
                                       %rest_command = [rest_command; 0];
                                 else
                                       %move_command = [move_command; 0];
                                       %rest_command = [rest_command; 1];
                                       valid_move_cnt = 0;           
                                       fprintf('.');
                                       kcount = kcount+1; 
                                       if kcount >= 30;
                                           kcount = 0;
                                           fprintf('\n');
                                       end
                                 end % endif
        
                                move_counts = [move_counts valid_move_cnt];
                                if  valid_move_cnt >= cloop_cnts_threshold;
                                     valid_move_cnt = 0;
                                end

                               end % end for sl_index = 1:window_length

           end % end if start_prediction == true
                                % Miscellaneous
%                                prev_pkt = new_pkt;
                                %prev_hp_filt_pkt = hp_filt_pkt;
                                %prev_lp_filt_pkt = lp_filt_pkt;
                                %prev_spatial_filt_pkt = spatial_filt_pkt;
%                                 prev_emg_bp_filt_pkt = emg_bp_filt_pkt;
%                                 prev_emg_abs_diff_pkt = emg_abs_diff_pkt;
%                                 prev_emg_mvc_pkt = emg_mvc_pkt;
                                
                                % unprocessed_eeg = [unprocessed_eeg new_pkt];  % test_change 8/22/2015
                                processed_eeg   = [processed_eeg Filtered_EEG_Downsamp];
                                Overall_spatial_chan_avg = [Overall_spatial_chan_avg spatial_chan_avg];
                            end   % end if dims_pkt > pkt_size

                            if handles.system.use_emg == 1 % If using EMG control only, then set eeg_control = 1
                               eeg_control = 1; 
                            end
                                
                            if eeg_control == 1                                 
                                    if ((emg_rms_pkt(emg_control_channels(1)) >= emg_mvc_threshold) || (emg_rms_pkt(emg_control_channels(2)) >= emg_tricep_threshold))            
                                            if handles.system.serial_interface.enable_serial
                                                    fwrite(handles.system.serial_interface.serial_object,[7]);                 
                                            elseif strcmp(get(hflexion_game.hball,'Visible'),'on') && strcmp(get(hflexion_game.move_ball_timer,'Running'),'off')
                                                    start(hflexion_game.move_ball_timer);
                                            else
                                                %ignore
                                            end          
                                             % commented 4-9-15
                                            %fprintf(handles.user.testing_data.emg_decisions_fileID,'%6.3f \t %6.3f \t 1 \t %d \n',...
                                             %   emg_rms_pkt(1), emg_rms_pkt(2), strcmp(get(handles.reset_eeg_control_timer,'Running'),'on')); 
                                            % marker_block = [marker_block; [datahdr.block*datahdr.points,400]];
                                            marker_block = [marker_block; [((datahdr.block - 1)*datahdr.points + 1),400]];      % Added 8/24/2015
                                            eeg_control = 0;
                                              disp('--->EEG+EMG GO');
                                   else
                                       disp('.');
                                        % commented 4-9-15
                                        %fprintf(handles.user.testing_data.emg_decisions_fileID,'%6.3f \t %6.3f \t %6.3f \t %6.3f \t 0 \t %d \n',...
                                        %  emg_rms_pkt(1), emg_rms_pkt(2), emg_mvc_threshold, emg_tricep_threshold, strcmp(get(handles.reset_eeg_control_timer,'Running'),'on')); 
                                    end
                            end
                                
                        case 3       
                %% Stop message   
                                disp('Stop');
                                data = pnet(con, 'read', hdr.size - header_size);
                                %finish = true;

                        otherwise
                %% ignore all unknown types, but read the package from buffer
                                data = pnet(con, 'read', hdr.size - header_size);
                    end % end Switch
                tryheader = pnet(con, 'read', header_size, 'byte', 'network', 'view', 'noblock');
                end % end while ~isempty(tryheader)
            catch err
                disp(err.message);
                % stop timer and close connection in case of an error
                stop(readEEGTimer);
                pnet('closeall');
            end % end try/catch
        %end % end Main loop

    %handles.user.calibration_data.block_num = handles.user.calibration_data.block_num + 1;
    %set(handles.textbox_block_num,'String',num2str(handles.user.calibration_data.block_num));
    guidata(hObject,handles);

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

% --- Executes on slider movement.
function slider_prob_threshold_Callback(hObject, eventdata, handles)
    global cloop_prob_threshold
    cloop_prob_threshold = get(hObject,'Value');
    set(handles.editbox_prob_threshold,'String',num2str(cloop_prob_threshold));
    guidata(hObject,handles);
    % --- Executes during object creation, after setting all properties.
    function slider_prob_threshold_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to slider_prob_threshold (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called

    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editbox_prob_threshold_Callback(hObject, eventdata, handles)
    % --- Executes during object creation, after setting all properties.
    function editbox_prob_threshold_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to editbox_prob_threshold (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called

    % Hint: edit controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

% --- Executes on slider movement.
function slider_counts_threshold_Callback(hObject, eventdata, handles)
    global cloop_cnts_threshold
    cloop_cnts_threshold = get(hObject,'value');
    set(handles.editbox_counts_threshold,'String',num2str(cloop_cnts_threshold));
    guidata(hObject,handles);
        % --- Executes during object creation, after setting all properties.
        function slider_counts_threshold_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to slider_counts_threshold (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called
        
        % Hint: slider controls usually have a light gray background.
        if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor',[.9 .9 .9]);
        end
        

function editbox_counts_threshold_Callback(hObject, eventdata, handles)
        % --- Executes during object creation, after setting all properties.
        function editbox_counts_threshold_CreateFcn(hObject, eventdata, handles)
        % hObject    handle to editbox_counts_threshold (see GCBO)
        % eventdata  reserved - to be defined in a future version of MATLAB
        % handles    empty - handles not created until after all CreateFcns called

        % Hint: edit controls usually have a white background on Windows.
        %       See ISPC and COMPUTER.
        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end
        
% --- Executes on slider movement.
function slider_emg_threshold_Callback(hObject, eventdata, handles)
    global emg_mvc_threshold
    emg_mvc_threshold = get(hObject,'value');
    set(handles.editbox_emg_threshold,'String',num2str(emg_mvc_threshold));
    guidata(hObject,handles);
        % --- Executes during object creation, after setting all properties.
        function slider_emg_threshold_CreateFcn(hObject, eventdata, handles)
        % Hint: slider controls usually have a light gray background.
        if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor',[.9 .9 .9]);
        end

function editbox_emg_threshold_Callback(hObject, eventdata, handles)
        % --- Executes during object creation, after setting all properties.
        function editbox_emg_threshold_CreateFcn(hObject, eventdata, handles)

        if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
            set(hObject,'BackgroundColor','white');
        end
    
        
% --- Executes on slider movement. %tricep
function slider_tricep_threshold_Callback(hObject, eventdata, handles)
global emg_tricep_threshold
    emg_tricep_threshold = get(hObject,'value');
    set(handles.editbox_triceps_threshold,'String',num2str(emg_tricep_threshold));
    guidata(hObject,handles); 
    % --- Executes during object creation, after setting all properties.
    function slider_tricep_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_tricep_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function editbox_triceps_threshold_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function editbox_triceps_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editbox_triceps_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function axes_raster_plot_CreateFcn(hObject, eventdata, handles)

function pushbutton_manual_save_data_Callback(hObject, eventdata, handles)
save_cloop_data(handles);
guidata(hObject,handles);    

function checkbox_auto_save_data_Callback(hObject, eventdata, handles)

function save_cloop_data(handles)
% Save variables
global processed_eeg processed_emg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold emg_rms_baseline

classifier_filename = handles.user.testing_data.classifier_filename;
var_filename = [handles.user.testing_data.closeloop_folder_path handles.user.calibration_data.subject_initials...
                '_ses' num2str(handles.user.testing_data.closeloop_sess_num) '_block' num2str(handles.user.testing_data.closeloop_block_num)...
                '_closeloop_results_' datestr(clock,'mm-dd-yyyy_HH_MM_SS') '.mat'];
save(var_filename,'classifier_filename','processed_eeg','processed_emg', 'Overall_spatial_chan_avg','all_feature_vectors','GO_Probabilities',...
    'move_counts','marker_block','all_cloop_prob_threshold','all_cloop_cnts_threshold','emg_rms_baseline');

function [vidobj, capture_fig] = start_video_capture(handles) % Unused function 8-11-2015
    % Create a figure window. This example turns off the default
    % toolbar, menubar, and figure numbering.
    capture_fig = figure('Toolbar','none', 'Menubar', 'none', 'NumberTitle','Off', 'Name','Video Capture Window','Position',[900 1100 400 300]);
    % Create the text label for the timestamp
    % hTextLabel = uicontrol('style','text','String','Timestamp', 'Units','normalized', 'Position',[0.85 -.04 .15 .08]);
    vidobj = -1;
    
    try
        vidobj = videoinput('winvideo', 1,'RGB24_800x600');
       
        % Adjusting brightness and exposure
        % https://makarandtapaswi.wordpress.com/2009/07/09/webcam-capture-in-matlab/#comments
        % Additional Link: http://www.mathworks.com/help/supportpkg/usbwebcams/ug/set-properties-for-webcam-acquisition.html
        % src = getselectedsource(vid); % get properties of video object
        % get(src) % gets the list of properties, also inspect(src)
        % propinfo(src, 'Brightness') % returns the range of acceptable values
        % set(src, 'Exposure', 1); % set specific properties

        video_filename = [handles.user.testing_data.closeloop_folder_path handles.user.calibration_data.subject_initials...
                '_ses' num2str(handles.user.testing_data.closeloop_sess_num) '_block' num2str(handles.user.testing_data.closeloop_block_num)...
                '_video.avi']; % create filename string
        aviobj = VideoWriter(video_filename);  % Create a |VideoWriter| object.
        aviobj.Quality = 50; 
        aviobj.FrameRate = 30;             % Defaukt source frame rate = 30

        vidobj.DiskLogger = aviobj; 
        vidobj.LoggingMode = 'disk';
        vidobj.TriggerRepeat = Inf;            %  If the FramesPerTrigger property is set to Inf, the object ignores the value of the TriggerRepeat property.
        vidobj.FramesPerTrigger = Inf;     % Record all frames
        %vidobj.FrameGrabInterval = 20;    % Default = 1. Grab one frame in every X frames. Higher the value less data gets logged
        
        vidRes = vidobj.VideoResolution;
        imWidth = vidRes(1);
        imHeight = vidRes(2);
        nBands = vidobj.NumberOfBands;
        hImage = image( zeros(imHeight, imWidth, nBands) );
        
        % Specify the size of the axes that contains the image object so that it displays the image at the right resolution and
        % centers it in the figure window.
        figSize = get(capture_fig,'Position');
        figWidth = figSize(3);
        figHeight = figSize(4);
        gca.unit = 'pixels';
        gca.position = [ ((figWidth - imWidth)/2) ((figHeight - imHeight)/2) imWidth imHeight ];
        
        % Set up the update preview window function.
        % setappdata(hImage,'UpdatePreviewWindowFcn',@mypreview_fcn);

        % Make handle to text label available to update function.
        % setappdata(hImage,'HandleToTimestampLabel',hTextLabel);

        % Display the video data in your GUI.
        preview(vidobj, hImage);
        start(vidobj);
        tic 
        % stop(vidobj);
        % stoppreview(vidobj);
        
    catch capture_err
        figure(capture_fig);
        axes;
        text(0.1, 1/2,capture_err.message,'FontSize',12,'Color','r');
        % delete(vidobj)
    end        
    
function stop_video_capture(handles)    % Unused function 8-11-2015
    stop(handles.user.testing_data.vidobj);
    stoppreview(handles.user.testing_data.vidobj);
    
    % When logging large amounts of data to disk, disk writing occasionally lags behind the acquisition. To determine whether all frames are written to disk, you can optionally use the DiskLoggerFrameCount property.
    while (handles.user.testing_data.vidobj.FramesAcquired ~= handles.user.testing_data.vidobj.DiskLoggerFrameCount) 
        pause(.1)
    end
    % You can verify that the FramesAcquired and DiskLoggerFrameCount properties have identical values by using these commands and comparing the output.
    % vidobj.FramesAcquired
    % vidobj.DiskLoggerFrameCount
%     time_elapsed = toc
    delete(handles.user.testing_data.vidobj);
    
% function mypreview_fcn(obj,event,himage)
% % Example update preview window function.
% 
% % Get timestamp for frame.
% tstampstr = event.Timestamp;
% 
% % Get handle to text label uicontrol.
% ht = getappdata(himage,'HandleToTimestampLabel');
% 
% % Set the value of the text label.
% ht.String = tstampstr;
% 
% % Display image data.
% himage.CData = event.Data

function checkbox_use_eeg_Callback(hObject, eventdata, handles)
handles.system.use_eeg = get(hObject,'Value');
 guidata(hObject,handles);
 
function checkbox_use_eeg_emg_Callback(hObject, eventdata, handles)
handles.system.use_eeg_emg = get(hObject,'Value');    
 guidata(hObject,handles);
 
function checkbox_use_emg_Callback(hObject, eventdata, handles)
handles.system.use_emg = get(hObject,'Value');    
 guidata(hObject,handles);
 
function checkbox_use_mahi_exo_Callback(hObject, eventdata, handles)
 handles.system.use_mahi_exo = get(hObject,'Value');    
 guidata(hObject,handles);
 
 function reset_eeg_control_timer_Callback(timer_object,eventdata,hObject)
     global eeg_control
     eeg_control = 0;
%      guidata(handles,hObject);

% --- Executes on button press in checkbox_left_hand_impaired.
function checkbox_left_hand_impaired_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_left_hand_impaired (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_left_hand_impaired
handles.system.left_impaired = get(hObject,'Value');
guidata(hObject,handles);

% --- Executes on button press in checkbox_right_hand_impaired.
function checkbox_right_hand_impaired_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_right_hand_impaired (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_right_hand_impaired
handles.system.right_impaired = get(hObject,'Value');
guidata(hObject,handles);
