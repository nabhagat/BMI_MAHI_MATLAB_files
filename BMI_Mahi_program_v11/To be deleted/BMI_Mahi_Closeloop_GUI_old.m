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

% Last Modified by GUIDE v2.5 11-Jul-2014 14:49:19

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

set(handles.textbox_subject_initials,'String',handles.user.calibration_data.subject_initials);
set(handles.textbox_session_num,'String',num2str(handles.user.calibration_data.sess_num));
set(handles.textbox_block_num,'String',num2str(handles.user.calibration_data.block_num));
set(handles.textbox_cond_num,'String',num2str(handles.user.calibration_data.cond_num));

set(handles.textbox_cloop_session_num,'String',num2str(handles.user.testing_data.closeloop_sess_num));
set(handles.textbox_cloop_block_num,'String',num2str(handles.user.testing_data.closeloop_block_num));

set(handles.textbox_comm_port_num,'String','--','Enable','off');
set(handles.slider_counts_threshold,'value',1);
set(handles.editbox_prob_threshold,'Enable','off');
set(handles.editbox_counts_threshold,'Enable','off');
handles.system.serial_interface.comm_port_num = 0;
handles.system.serial_interface.enable_serial = 0;
handles.system.serial_interface.serial_object = varargin{2};

set(handles.axes_raster_plot,'box','on','nextplot','replacechildren');

% Start flexion_game gui
eval('flexion_game');
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
if ishandle(hflexion_game.figure_flexion_game) && strcmp(get(hflexion_game.figure_flexion_game, 'type'), 'figure')
    % findobj('type','figure','name','mytitle')
    close(hflexion_game.figure_flexion_game);
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

global processed_eeg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold

varargout{1} = processed_eeg;
varargout{2} = Overall_spatial_chan_avg;
varargout{3} = all_feature_vectors;
varargout{4} = GO_Probabilities;
varargout{5} = move_counts;
varargout{6} = marker_block;
varargout{7} = all_cloop_prob_threshold;
varargout{8} = all_cloop_cnts_threshold;


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
    global resamp_Fs
    global window_length
    global window_time
    global CloopClassifier
    global Best_BMI_classifier
    global cloop_prob_threshold
    global cloop_cnts_threshold
    global DEFINE_GO
    global DEFINE_NOGO
    global target_trig_label
    global move_trig_label
    global rest_trig_label
    global block_start_stop_label
    
    % Load Performance variable from .mat file
    handles.user.calibration_data.folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' handles.user.calibration_data.subject_initials '\' ...
                                                handles.user.calibration_data.subject_initials '_Session'...
                                                num2str(handles.user.calibration_data.sess_num) '\'];   %change7
                                            
    handles.user.testing_data.closeloop_folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' handles.user.calibration_data.subject_initials '\' ...
                                                handles.user.calibration_data.subject_initials '_Session'...
                                                num2str(handles.user.testing_data.closeloop_sess_num) '\'];     %change8
    
    handles.user.testing_data.classifier_filename = [handles.user.calibration_data.folder_path...
                 handles.user.calibration_data.subject_initials...
                 '_ses' num2str(handles.user.calibration_data.sess_num)...
                 '_cond' num2str(handles.user.calibration_data.cond_num)...
                 '_block' num2str(handles.user.calibration_data.block_num)...
                 '_performance_optimized_causal.mat'];
             
    load(handles.user.testing_data.classifier_filename);
    CloopClassifier = Performance;
    classifier_channels = CloopClassifier.classchannels;
    %classifier_channels = [9 10 16 17];     % change9

    resamp_Fs = 20; % required Sampling Freq
    window_length = CloopClassifier.smart_window_length*resamp_Fs + 1; % No. of samples required, Earlier  = length(CloopClassifier.move_window(1):1/resamp_Fs:CloopClassifier.move_window(2));   
    window_time = 1/resamp_Fs:1/resamp_Fs:window_length/resamp_Fs;

    % Decide classifier with best accuracy
    [max_acc_val,max_acc_index] = max(CloopClassifier.eeg_accur); 
    Best_BMI_classifier = CloopClassifier.eeg_svm_model{max_acc_index};

    % Classifier parameters
    cloop_prob_threshold = CloopClassifier.opt_prob_threshold;            
    cloop_cnts_threshold  = CloopClassifier.consecutive_cnts_threshold;        % Number of consecutive valid cnts required to accept as 'GO'

    DEFINE_GO = 1;
    DEFINE_NOGO = 2;
    target_trig_label = 'S  8';
    move_trig_label = 'S 16';  % 'S 32'; %'S  8'; %'100';   %change10
    rest_trig_label = 'S  2';  % 'S  2'; %'200';
    block_start_stop_label = 'S 10';
    
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
    
    % Define and start timer for reading from socket every 100 msec
    readEEGTimer = timer('TimerFcn', {@EEGTimer_Callback,hObject}, 'Period', 0.01, 'ExecutionMode','fixedRate'); 
    start(readEEGTimer);
    
    set(handles.slider_prob_threshold,'value',cloop_prob_threshold);
    set(handles.editbox_prob_threshold,'String',num2str(cloop_prob_threshold));
    set(handles.slider_counts_threshold,'value',cloop_cnts_threshold);
    set(handles.editbox_counts_threshold,'String',num2str(cloop_cnts_threshold));
    
    handles.user.testing_data.closeloop_block_num = handles.user.testing_data.closeloop_block_num + 1;
    set(handles.textbox_cloop_block_num,'String',num2str(handles.user.testing_data.closeloop_block_num));
    guidata(hObject,handles);

function pushbutton_stop_closeloop_Callback(hObject, eventdata, handles)
% global definitions
    global processed_eeg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
    global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold datahdr
    
    global readEEGTimer;       % Timer object   
    % Stop the timer
    stop(readEEGTimer);
    % Close all open socket connections
    pnet('closeall');
    % Display a message
    disp('connection closed');
    
    %% Raster plot for close loop 
try       
                    Proc_EEG = processed_eeg; 
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
                    pred_move_onset_times = round(event_times(Markers(:,2)==100).*proc_Fs)/proc_Fs;

                    pred_Ovr_Spatial_Avg = Ovr_Spatial_Avg(find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2))); % Correct the '-7'
                    pred_eeg_time = eeg_time(find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2)));
                    pred_Proc_EEG = Proc_EEG(:,find(eeg_time == pred_start_stop_times(1)):find(eeg_time == pred_start_stop_times(2)));

                    if length(All_Feature_Vec) > length(pred_Ovr_Spatial_Avg)
                        All_Feature_Vec = All_Feature_Vec(:,1:length(pred_Ovr_Spatial_Avg));
                        GO_Prob = GO_Prob(1:length(pred_Ovr_Spatial_Avg));
                        Num_Move_Counts = Num_Move_Counts(1:length(pred_Ovr_Spatial_Avg));
                    else
                        pred_Ovr_Spatial_Avg = pred_Ovr_Spatial_Avg(1:length(All_Feature_Vec));
                        pred_eeg_time = pred_eeg_time(1:length(All_Feature_Vec));
                        pred_Proc_EEG = pred_Proc_EEG(:,1:length(All_Feature_Vec));
                    end

                    raster_data = [All_Feature_Vec(1:3,1:length(pred_Ovr_Spatial_Avg)); [zeros(1,50) All_Feature_Vec(4,51:length(pred_Ovr_Spatial_Avg))]; pred_Ovr_Spatial_Avg]'; % Needs to be corrected
                    %raster_zscore = zscore(raster_data);
                    raster_zscore = raster_data;
                    raster_time = pred_eeg_time;
                    raster_colors = ['m','k','b','r','k'];
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
                    %plot(raster_time, 10/3*Num_Move_Counts, '--b','LineWidth',2);
                    hold on;
                    plot(handles.axes_raster_plot,raster_time, 10 + 10*GO_Prob, '.r','LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, 10 + 10*all_cloop_prob_threshold(1:length(pred_Ovr_Spatial_Avg)),'-','Color',[0.4 0.4 0.4],'LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, all_cloop_cnts_threshold(1:length(pred_Ovr_Spatial_Avg)),'-k','LineWidth',2);
                    plot(handles.axes_raster_plot,raster_time, Num_Move_Counts,'b');
                    %line([0 200],10.*[cloop_prob_threshold cloop_prob_threshold],'Color',[0.4 0.4 0.4],'LineWidth',2);
                    axis(handles.axes_raster_plot,[pred_start_stop_times(1) pred_start_stop_times(2) 1 50]);
                    %ylim([0 40]);

                    myaxis = axis;
                    for plot_ind1 = 1:length(pred_stimulus_times);
                        line([pred_stimulus_times(plot_ind1), pred_stimulus_times(plot_ind1)],[myaxis(3)-1, myaxis(4)],'Color','b','LineWidth',1);
                        text(pred_stimulus_times(plot_ind1)-1,myaxis(4)+0.5,'Start','Rotation',60,'FontSize',12);
                        %text(pred_stimulus_times(plot_ind1)-1.5,myaxis(4)+1,'shown','Rotation',0,'FontSize',12);
                        hold on;
                    end

                    for plot_ind2 = 1:length(pred_GO_times);
                        line([pred_GO_times(plot_ind2), pred_GO_times(plot_ind2)],[myaxis(3)-1, myaxis(4)],'Color','k','LineWidth',1,'LineStyle','--');
                        text(pred_GO_times(plot_ind2),myaxis(4)+0.5,'GO','Rotation',60,'FontSize',12);
                        hold on;
                    end

                    for plot_ind3 = 1:length(pred_move_onset_times);
                        line([pred_move_onset_times(plot_ind3), pred_move_onset_times(plot_ind3)],[myaxis(3), myaxis(4)],'Color','r','LineWidth',2,'LineStyle','--');
                        hold on;
                    end
                    set(handles.axes_raster_plot,'YTick',[1 all_cloop_cnts_threshold(1) 5 10 10+10*all_cloop_prob_threshold(1) 20 25 30 35 40 45]);
                    set(handles.axes_raster_plot,'YTickLabel',{'1' 'Count_Thr' '5' 'p(GO) = 0','Prob_Thr','p(GO) = 1', 'AUC', '-ve Peak','Slope','Mahal','Spatial Avg'});
                    xlabel(handles.axes_raster_plot,'Time (sec.)', 'FontSize' ,12);
                    %title(handles.axes_raster_plot,'Raster Plot','FontSize',12);
                    %export_fig 'Block8_results' '-png' '-transparent'
                    hold off;    
catch raster_err
        cla(handles.axes_raster_plot,'reset')    
        axes_xlim = get(handles.axes_raster_plot,'XLim');
        axes_ylim = get(handles.axes_raster_plot,'YLim');
        text(axes_xlim(1) + 0.2, axes_ylim(2)/2,'Error: Insufficient data to create plot','FontSize',12,'Color','r');
        text(axes_xlim(1) + 0.2, axes_ylim(2)/2-0.2, raster_err.message,'Color','r');
end

    if get(handles.checkbox_auto_save_data,'Value')
        save_cloop_data(handles);
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
    global resamp_Fs
    global window_length
    global window_time
    global CloopClassifier
    global Best_BMI_classifier
    global cloop_prob_threshold
    global cloop_cnts_threshold
    global DEFINE_GO
    global DEFINE_NOGO
    global target_trig_label
    global move_trig_label
    global rest_trig_label
    global block_start_stop_label
    global hflexion_game
    global datahdr
    
    % 2. Variables initialized in EEGTimer_Callback()
    global orig_Fs
    global lastBlock;       % Number of last block for overflow test
    global props;           % EEG Properties
    global dataX00ms;          % EEG data of the last recorded second
    global total_chnns
    global downsamp_factor pkt_size filt_order
    global new_pkt prev_pkt hp_filt_pkt lp_filt_pkt spatial_filt_pkt
    global firstblock marker_block marker_type
    global processed_eeg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
    global valid_move_cnt move_counts feature_matrix sliding_window start_prediction kcount
    global all_cloop_prob_threshold all_cloop_cnts_threshold
    global required_chnns L_LAP M_CAR
       
    
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
                            total_chnns = double(props.channelCount);
                            new_pkt = zeros(total_chnns,pkt_size);
                            prev_pkt = zeros(total_chnns,pkt_size);
                            hp_filt_pkt = zeros(total_chnns,pkt_size);
                            lp_filt_pkt = zeros(length(classifier_channels),pkt_size);      % Use only classifier channels
                            spatial_filt_pkt = zeros(total_chnns,pkt_size);

                            firstblock = true;          % First block of data read in 
                            marker_block = [];
                            marker_type = [];

                            processed_eeg = [];
                            %unprocessed_eeg = [];
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
                                marker_block = [marker_block; [datahdr.block*datahdr.points,50]];      % Not completely accurate
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
                                       %marker_block = [marker_block; [markers(m).sample_num,200]];
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
                                            if handles.system.serial_interface.enable_serial
                                                fwrite(handles.system.serial_interface.serial_object,[7]);                
                                            elseif strcmp(get(hflexion_game.hball,'Visible'),'on') && strcmp(get(hflexion_game.move_ball_timer,'Running'),'off')
                                                start(hflexion_game.move_ball_timer);
                                            else
                                                %ignore
                                            end                                                  
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

function axes_raster_plot_CreateFcn(hObject, eventdata, handles)

function pushbutton_manual_save_data_Callback(hObject, eventdata, handles)
save_cloop_data(handles);
guidata(hObject,handles);    

function checkbox_auto_save_data_Callback(hObject, eventdata, handles)

function save_cloop_data(handles)
% Save variables
global processed_eeg Overall_spatial_chan_avg all_feature_vectors GO_Probabilities
global move_counts marker_block all_cloop_prob_threshold all_cloop_cnts_threshold

classifier_filename = handles.user.testing_data.classifier_filename;
var_filename = [handles.user.testing_data.closeloop_folder_path handles.user.calibration_data.subject_initials...
                '_ses' num2str(handles.user.testing_data.closeloop_sess_num) '_block' num2str(handles.user.testing_data.closeloop_block_num)...
                '_closeloop_results.mat'];
save(var_filename,'classifier_filename','processed_eeg','Overall_spatial_chan_avg','all_feature_vectors','GO_Probabilities',...
    'move_counts','marker_block','all_cloop_prob_threshold','all_cloop_cnts_threshold');
    
    
        
