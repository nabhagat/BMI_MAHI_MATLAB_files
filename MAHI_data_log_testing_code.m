%% BMI Exo's data analysis
% James French
% Jannuary 2014

%% Import data
% clc; clear;

[fileName,pathName] = uigetfile('*.txt');
fileID = fopen([pathName fileName]);
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f';  %floating point number
data = textscan(fileID,formatSpec,'HeaderLines',14,'MultipleDelimsAsOne',1);
fclose(fileID);

time = data{1,1};
position_elbow = data{1,2};     %rad
position_forearm = data{1,3};
position_wrist1 = data{1,4};       %m
position_wrist2 = data{1,5};
position_wrist3 = data{1,6};

%rad to deg
position_elbow = (180/pi).*(position_elbow);
position_forearm = (180/pi).*(position_forearm);

velocity_elbow = data{1,7};    %rad/s
velocity_forearm = data{1,8};
velocity_wrist1 = data{1,9};       %m/s
velocity_wrist2 = data{1,10};
velocity_wrist3 = data{1,11};

%rad to deg
velocity_elbow = (180/pi).*(velocity_elbow);
velocity_forearm = (180/pi).*(velocity_forearm);

torque_elbow = data{1,12};    %percentage
torque_forearm = data{1,13};
torque_wrist1 = data{1,14};
torque_wrist2 = data{1,15};
torque_wrist3 = data{1,16};

trigger1 = data{1,17};
trigger2 = data{1,18};
trigger_mov = data{1,19};
target_num = data{1,20};
count = data{1,21};

%% Plot all positions and velocities in a giant figure (subplot)
fontsize = 10;

figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,5,1), plot(time,position_elbow,'.b')
    title('ELBOW Position')
    ylabel('deg','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,6), plot(time,velocity_elbow,'.r')
    title('Velocity')
    ylabel('deg/s','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,11), plot(time,torque_elbow,'.m')
    title('Torque at Motor')
    ylabel('Nm','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
    
subplot(3,5,2), plot(time,position_forearm,'.b')
    title('FOREARM Position')
    ylabel('deg','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,7), plot(time,velocity_forearm,'.r')
    title('Velocity')
    ylabel('deg/s','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,12), plot(time,torque_forearm,'.m')
    title('Torque at Motor')
    ylabel('Nm','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
    
subplot(3,5,3), plot(time,position_wrist1,'.b')
    title('wrist1 Position (motor #3)')
    ylabel('m','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,8), plot(time,velocity_wrist1,'.r')
    title('Velocity')
    ylabel('m/s','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,13), plot(time,torque_wrist1,'.m')
    title('Torque at Motor')
    ylabel('Nm','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);

subplot(3,5,4), plot(time,position_wrist2,'.b')
    title('wrist2 Position (motor #4)')
    ylabel('m','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,9), plot(time,velocity_wrist2,'.r')
    title('Velocity')
    ylabel('m/s','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,14), plot(time,torque_wrist2,'.m')
    title('Torque at Motor')
    ylabel('Nm','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);

subplot(3,5,5), plot(time,position_wrist3,'.b')
    title('wrist3 Position (motor #5)')
    ylabel('m','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,10), plot(time,velocity_wrist3,'.r')
    title('Velocity')
    ylabel('m/s','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,5,15), plot(time,torque_wrist3,'.m')
    title('Torque at Motor')
    ylabel('Nm','fontsize',fontsize)
    grid on
    set(gca,'fontsize',fontsize);
   
%Super title
% [ax1,h1]=sublabel(fileName  ,'t');
% set(h1,'fontsize',22,'interpreter','none')

figure
subplot(3,1,1), plot(time,trigger1,'b','linewidth',2)
    title('Trigger1 - New Target Appears')
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,1,2), plot(time,trigger2,'r','linewidth',2)
    title('Trigger2 - Target Reached')
    grid on
    set(gca,'fontsize',fontsize);
subplot(3,1,3), plot(time,target_num,'k','linewidth',2)
    title('Target Number: (0=upper and lower target, 1=lower, 2=center, 3=upper, 4=none)')
    grid on
    set(gca,'fontsize',fontsize);
    
%% Plot elbow position and velocity and trigger

fontsize = 10;

figure
title('elbow - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_elbow,'.b')
plot(time,velocity_elbow,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);
hold on
plot(time,trigger_mov,'k','linewidth',2)
    
    
    
%% Plot all positions and velocities in individual figures
fontsize = 10;

figure
title('elbow - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_elbow,'.b')
plot(time,velocity_elbow,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);

figure
title('forearm - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_forearm,'.b')
plot(time,velocity_forearm,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);

figure
title('wrist1 - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_wrist1,'.b')
plot(time,velocity_wrist1,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);

figure
title('wrist2 - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_wrist2,'.b')
plot(time,velocity_wrist2,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);

figure
title('wrist3 - Position and Velocity','fontsize',fontsize), hold on
plot(time,position_wrist3,'.b')
plot(time,velocity_wrist3,'.r')
legend('Position','Velocity')
xlabel('Time [s]','fontsize',fontsize)
ylabel('deg and deg/s','fontsize',fontsize)
grid on
set(gca,'fontsize',fontsize);

