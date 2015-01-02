% Program to simulate Real-time processing 
% Create by Nikunj Bhagat, University of Houston
% Date: 16-9-2013 (DD-MM-YYYY)
clear;
%close all;

% Create a sine wave
Fs = 500;
Ts = 1/Fs;
t = (0:Ts:100-(Ts));
Fc = 0.5; 
%Sine_t = [0.1*sin(2*pi*Fc*t)+0.1*sin(2*pi*100*Fc*t)+10; 0.05*cos(2*pi*Fc*t)];
Sine_t = [0.1*sin(2*pi*Fc*t); 0.05*cos(2*pi*Fc*t)];

% plot(t,Sine_t);

% Extract pkts 
pkt_size = 250;
[n_channels,n_samples] = size(Sine_t);
filt_order = 4;
new_pkt = zeros(n_channels,pkt_size);
prev_pkt = zeros(n_channels,pkt_size);
hp_filt_pkt = zeros(n_channels,pkt_size);
lp_filt_pkt = zeros(n_channels,pkt_size);
car = zeros(n_channels,pkt_size);
processed_signal = [];
unprocessed_signal = [];

window_size = 5;        % 5*100ms = 500 ms, Number of feature per channel
curr_window = zeros(n_channels,window_size);
sliding_window = zeros(n_channels,window_size);

% Main loop
for pkt_cnt = 1:n_samples/pkt_size
    ra1 = (pkt_cnt-1)*pkt_size+1;
    ra2 = pkt_cnt*pkt_size;
    new_pkt = Sine_t(:,ra1:ra2);
    % High pass filter 0.1 Hz
    for no_chns = 1:n_channels
             hp_filt_pkt(no_chns,:) = pkt_hp_filter(new_pkt(no_chns,:),prev_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size),hp_filt_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size));
    end
    % Re-reference the data
    avg_ref = mean(hp_filt_pkt,1);
    for no_chns = 1:n_channels
        car(no_chns,:) = hp_filt_pkt(no_chns,:) - avg_ref;
    end
    % Low pass filter 1 Hz
    for no_chns = 1:n_channels
             lp_filt_pkt(no_chns,:) = pkt_lp_filter(car(no_chns,:),prev_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size),lp_filt_pkt(no_chns,(pkt_size - filt_order)+1:pkt_size));
    end
    
    % Downsample data to 10 Hz
    for no_chns = 1:n_channels
        downsamp_pkt(no_chns,:) = downsample(lp_filt_pkt(no_chns,:),50);
    end
    
    % Implement Sliding Window
    curr_window = downsamp_pkt;
    for sl_index = 1:window_size
        sliding_window = [sliding_window(:,2:window_size) curr_window(:,sl_index)];
        
        % % Old Logic
        %if sl_index < window_size
            %sliding_window = [prev_window(:,sl_index+1:window_size) curr_window(:,1:sl_index)];
        %else
            %sliding_window = curr_window(:,1:sl_index);
            %prev_window = curr_window;
        %end
        
        % Make a predict
    end
    
    % Miscellaneous
    prev_pkt = new_pkt;
    processed_signal = [processed_signal downsamp_pkt];
    unprocessed_signal = [unprocessed_signal prev_pkt];
    
end

figure;
plot(t,Sine_t(1,:),'b',t,Sine_t(2,:),'r');
%plot(t,Sine_t(1,:),'b');
hold on; plot(t,processed_signal(1,:),'g',t,processed_signal(2,:),'k');
%hold on; plot(t,processed_signal(1,:),'r');
hold off;



