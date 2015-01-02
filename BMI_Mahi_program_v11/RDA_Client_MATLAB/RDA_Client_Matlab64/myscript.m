% myscript
clear;

[proc_eeg,unproc_eeg,no_move,no_rest,markers] = RDA();

markers = double(markers);
markers(:,1) = markers(:,1) - markers(1,1);
markers(:,1) = markers(:,1)*5/500;
% Round upto 1 decimal place
markers(:,1) = floor(markers(:,1)*10)/10;

%% Generate Raster Plot

    EEG.data = proc_eeg; Fs = 10;
    %EEG.data = unproc_eeg; Fs = 500;
    Ts = 1/Fs;
    Channels_nos = [38;5;39;9;10;48;14;49;19;53;20];
    raster_colors = ['g','r','b','k','y','c','m','g','r','b','k','r','b','k','g','r','b','k','y','c','m'];
    raster_Fs = Fs;
    raster_lim1 = 1;
    raster_lim2 = length(EEG.data);
    raster_time = (0:Ts:(length(EEG.data)-1)*(Ts));
    [no_channels,no_samples] = size(EEG.data);
    no_channels = length(Channels_nos);
    
    % Plot selected EEG channels and kinematics
    raster_data = zeros(length(EEG.data(1,raster_lim1:raster_lim2)),no_channels);
    for chan_index = 1:no_channels
        raster_data(:,chan_index) = EEG.data(Channels_nos(chan_index),:)';
    end
    raster_data = [raster_data no_move no_rest];
    
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
    
    axis([raster_lim1/raster_Fs raster_lim2/raster_Fs raster_ylim1 raster_ylim2]);
    set(gca,'YTick',[5 10 15 20 25 30 35 40 45 50 55 60 65 70]);
    set(gca,'YTickLabel',{'Fz','FC1','FC2','Cz','CP1','CP2','F1','F2','C1','C2','CPz','Move','Rest','Tang Velocity'},'FontSize',8);
    %set(gca,'XTick',[kin_stimulus_trig(10)/raster_Fs kin_response_trig(11)/raster_Fs]);
    %set(gca,'XTickLabel',{'Target','Movement'});
    xlabel('Time (sec.)','FontSize',10);
    hold off;

    %Plot Markers on the raster plot
    for plot_ind1 = 2:length(markers);
        if markers(plot_ind1,2) == 100
            line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','r');
        elseif markers(plot_ind1,2) == 200
            line([markers(plot_ind1,1) markers(plot_ind1,1)],[0 100],'Color','g');
        end
    hold on;
    end
    