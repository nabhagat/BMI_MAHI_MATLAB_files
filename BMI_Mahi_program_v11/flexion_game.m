function varargout = flexion_game(varargin)
% FLEXION_GAME MATLAB code for flexion_game.fig
%      FLEXION_GAME, by itself, creates a new FLEXION_GAME or raises the existing
%      singleton*.
%
%      H = FLEXION_GAME returns the handle to a new FLEXION_GAME or the handle to
%      the existing singleton*.
%
%      FLEXION_GAME('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FLEXION_GAME.M with the given input arguments.
%
%      FLEXION_GAME('Property','Value',...) creates a new FLEXION_GAME or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before flexion_game_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to flexion_game_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help flexion_game

% Last Modified by GUIDE v2.5 10-Jul-2014 21:47:44

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @flexion_game_OpeningFcn, ...
                   'gui_OutputFcn',  @flexion_game_OutputFcn, ...
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


% --- Executes just before flexion_game is made visible.
function flexion_game_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to flexion_game (see VARARGIN)

% Choose default command line output for flexion_game
handles.output = hObject;
global hflexion_game
global home_ball_position home_gap_position
global datahdr marker_block
marker_block = [];
datahdr.block = 0;
datahdr.points = 0;
datahdr.markerCount = 0;


screen_size = get(0,'ScreenSize');
set(handles.figure_flexion_game,'Menubar', 'none',...
                                'Name','Flexi Game',...
                                'WindowStyle','normal',...
                                'Position',[screen_size(3)/2-400 screen_size(4)/2-400 800 800]); 

set(handles.axes_flexion_game,'box','on',...
                              'color',[0.6 0.6 0.6],...
                              'Position',[0 0 1 1],...
                              'XLim',[0 10],...
                              'YLim',[0 10]);

handles.hfixation = draw_circle_cross(handles.axes_flexion_game,5,5,2,0.3,'k');
set(handles.hfixation,'Visible','on');
handles.hfence = rectangle('Position',[0 6 10 0.5],'Curvature',[0 0],'FaceColor','r','EdgeColor','none'); 
handles.hgap = rectangle('Position',[-3 6 10 0.5],'Curvature',[0 0],...
               'FaceColor',get(handles.axes_flexion_game,'Color'),...
                'EdgeColor','none');        % [-3 6 4 0.5]
            
home_gap_position = get(handles.hgap,'Position');
handles.htarget = draw_circle_cross(handles.axes_flexion_game,5,9,0.8,0.15,'k');
handles.hball = draw_circle_cross(handles.axes_flexion_game,5,1,0.8,0.15,'g');
handles.hball = handles.hball(1); % Use only outer circle :)
home_ball_position = get(handles.hball,'Position');

set(handles.htarget,'Visible','off');
set(handles.hball,'Visible','off');
set(handles.hfence,'Visible','off');
set(handles.hgap,'Visible','off');
set(handles.pushbutton_GO,'Visible','on');

handles.target_appears_timer = timer('TimerFcn', {@target_appears_timer_Callback,hObject},...
                                     'StartDelay', 5, 'Period', 5,'TasksToExecute',1,...
                                     'ExecutionMode','fixedRate');
                                 
handles.move_ball_timer = timer('TimerFcn', {@move_ball_timer_Callback,hObject},...
                                     'StartDelay', 0, 'Period', 0.05,'ExecutionMode','fixedRate');
                                 
handles.move_gap_timer = timer('TimerFcn', {@move_gap_timer_Callback,hObject},...
                                     'StartDelay', 0, 'Period', 0.01,'ExecutionMode','fixedRate');  % chnage 0.1
                                 
start(handles.target_appears_timer);

hflexion_game = handles;
% Update handles structure
guidata(hObject, handles); 

% UIWAIT makes flexion_game wait for user response (see UIRESUME)
% uiwait(handles.figure_flexion_game);


% --- Outputs from this function are returned to the command line.
function varargout = flexion_game_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton_GO.
function pushbutton_GO_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_GO (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
start(handles.move_ball_timer);
guidata(hObject,handles);

function target_appears_timer_Callback(timer_object,eventdata,hObject)
global home_ball_position home_gap_position
global datahdr marker_block
handles = guidata(hObject);
stop(handles.target_appears_timer);
% randomly wait additional 2 - 4 sec. 
% pause(2+(4-2)*rand);
set(handles.hfixation,'Visible','off');

set(handles.hfence,'Visible','off'); % Change to hide/show fence
set(handles.hgap,'Visible','on');
set(handles.hgap,'Position',home_gap_position + [1 + (10-1)*rand 0 0 0]);
start(handles.move_gap_timer);

set(handles.htarget,'Visible','on');
marker_block = [marker_block; [datahdr.block*datahdr.points,200]];      % Not completely accurate. Target Appears
set(handles.hball,'Position',home_ball_position);
set(handles.hball,'Visible','on');
guidata(hObject,handles);

function move_gap_timer_Callback(timer_object,eventdata,hObject)
handles = guidata(hObject);
global home_gap_position
gap_pos = get(handles.hgap,'Position');
axes_xlim = get(handles.axes_flexion_game,'Xlim'); 
if gap_pos(1) >= axes_xlim(2)-1
   set(handles.hgap,'Position',home_gap_position);
else
   gap_pos(1) = gap_pos(1) + 0.1 + (0.5-0.1)*rand;        % random increment [0.1 0.5]
   %gap_pos(3) = gap_pos(3) + 2 + (4-2)*rand;        % random gap width increment [2 6]
   set(handles.hgap,'Position',gap_pos);
end

function move_ball_timer_Callback(timer_object,eventdata,hObject)
handles = guidata(hObject);
ball_pos = get(handles.hball,'Position');
target_pos = get(handles.htarget,'Position');
gap_pos = get(handles.hgap,'Position');

if abs((ball_pos(2) + ball_pos(4)/2) - gap_pos(2)) < 0.5 && ...
    ~(ball_pos(1)+0.5 > gap_pos(1) && ((ball_pos(1) + ball_pos(3)-0.5) < (gap_pos(1) + gap_pos(3))))
        %disp('Ouchh...');
        ball_pos(2) = ball_pos(2);  %hold current ball position
        set(handles.hball,'Position',ball_pos);
elseif ball_pos(2) >=  target_pos{1,1}(2)
    stop(handles.move_ball_timer);
    stop(handles.move_gap_timer);
    
    set(handles.hball,'Visible','off');
    set(handles.htarget,'Visible','off');
    set(handles.hfence,'Visible','off');
    set(handles.hgap,'Visible','off');
    set(handles.hfixation,'Visible','on');
    start(handles.target_appears_timer);
else
    ball_pos(2) = ball_pos(2) +  0.3;
    set(handles.hball,'Position',ball_pos);
end
guidata(hObject,handles);

function handle_circle_cross = draw_circle_cross(current_axes_handle,center_x, center_y, Radius,width,color)
% Function to create circle with cross for Flexion_Game GUI
% Created by: Nikunj A. Bhagat, University of Houston
% Date: July 10th, 2014

axes_xlim = get(current_axes_handle,'XLim');
axes_ylim = get(current_axes_handle,'YLim');

outer_x = center_x - Radius - axes_xlim(1);
outer_y = center_y - Radius - axes_ylim(1);
outer_w = 2*Radius;
outer_h = 2*Radius;
outer_circle = rectangle('Position',[outer_x outer_y outer_w outer_h],...
                         'Curvature',[1 1], 'FaceColor',color,'Visible','off');

inner_x = outer_x + width;
inner_y = outer_y + width;
inner_w = 2*(Radius - width);
inner_h = 2*(Radius - width);
inner_circle = rectangle('Position',[inner_x inner_y inner_w inner_h],...
                         'Curvature',[1 1], 'FaceColor',get(current_axes_handle,'color'),'Visible','off');

vertical_x = center_x - width/2;
vertical_y = center_y - Radius + width/2;
vertical_w = width;
vertical_h = 2*Radius - width;
vertical_bar = rectangle('Position',[vertical_x vertical_y vertical_w vertical_h],...
                         'Curvature',[0 0], 'FaceColor',color,'Visible','off');
                     
horizontal_x = center_x - Radius + width/2;
horizontal_y = center_y - width/2;
horizontal_w = 2*Radius - width;
horizontal_h = width;
horizontal_bar = rectangle('Position',[horizontal_x horizontal_y horizontal_w horizontal_h],...
                         'Curvature',[0 0], 'FaceColor',color,'Visible','off');
                     
handle_circle_cross = [outer_circle inner_circle vertical_bar horizontal_bar]; 

% --- Executes when user attempts to close figure_flexion_game.
function figure_flexion_game_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure_flexion_game (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

stop(handles.target_appears_timer);
stop(handles.move_ball_timer);
stop(handles.move_gap_timer);

delete(handles.target_appears_timer);
delete(handles.move_ball_timer);
delete(handles.move_gap_timer);

% Hint: delete(hObject) closes the figure
delete(hObject);
