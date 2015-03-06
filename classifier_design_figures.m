% Plot figures for classifier design steps
clear;
paper_font_size = 10;
% Select figures to plot
plot_erp_image = 0; 
plot_electrode_locations = 0;
plot_scalp_map = 0;
plot_eeg_emg_latencies = 0;
plot_feature_space = 0;
plot_offline_performance = 0;
plot_offline_performance_new = 0;
plot_single_spatial_average_trials = 0;
plot_classifier_design_all = 0;

% Subject Details
Subject_name = 'BNBO';
Sess_num = '2';
Cond_num = 3;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;

folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_average_causal.mat']);      % Always use causal for training classifier
Performance = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
Performance_conv = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 
EEG = pop_loadset( [Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_move_epochs.set'], folder_path);

Fs_eeg = 20; % Desired sampling frequency in real-time
downsamp_factor = Average.Fs_eeg/Fs_eeg;
move_epochs = Average.move_epochs;
rest_epochs = Average.rest_epochs;
[no_epochs,no_datapts,no_channels] = size(Average.move_epochs); %#ok<*ASGLU>

%-. Use separate move and rest epochs? (Always yes). 0 - No; 1 - yes
use_separate_move_rest_epochs = 1; 

% Downsample the epochs and epoch_times
    for k = 1:no_channels
        move_epochs_s(:,:,k) = downsample(move_epochs(:,:,k)',downsamp_factor)';
        if use_separate_move_rest_epochs == 1
            rest_epochs_s(:,:,k) = downsample(rest_epochs(:,:,k)',downsamp_factor)';
        else 
            rest_epochs_s(:,:,k) = downsample(move_epochs(:,:,k)',downsamp_factor)';
        end
    end
    [no_epochs,no_datapts,no_channels] = size(move_epochs_s);
    move_erp_time = downsample(Average.move_erp_time(:),downsamp_factor);
    if use_separate_move_rest_epochs == 1
        rest_erp_time = downsample(Average.rest_erp_time,downsamp_factor);
    else 
        rest_erp_time = downsample(Average.move_erp_time,downsamp_factor);
    end

    classchannels = Performance.classchannels;
    MRCP_chans = Average.RP_chans;
    grand_average_chans = [43, 9, 32, 10, 44, 13, 48, 14, 49, 15, 52, 19, 53, 20, 54];
    
   % Find peak within interval, Reject trials reject_trial_onwards 
    find_peak_interval = [-2.0 0.5];            % ranges up to 0.5 sec after movement onset because of phase delay filtering
    reject_trial_before = -1.5; % Seconds. reject trials for which negative peak is reached earlier than -1.5 sec
    
    bad_move_trials = []; 
    good_move_trials = [];
    move_ch_avg_ini = mean(move_epochs_s(:,:,classchannels),3);
    rest_ch_avg_ini = mean(rest_epochs_s(:,:,classchannels),3);

    mt1 = find(move_erp_time == find_peak_interval(1));
    mt2 = find(move_erp_time == find_peak_interval(2));
    [min_avg(:,1),min_avg(:,2)] = min(move_ch_avg_ini(:,mt1:mt2),[],2); % value, indices

    for nt = 1:size(move_ch_avg_ini,1)
        %if (move_erp_time(move_ch_avg_ini(nt,mt1:mt2) == min_avg(nt,1)) <= reject_trial_before) %|| (min_avg(nt,1) > -3)        
        if (move_erp_time(mt1 + find(move_ch_avg_ini(nt,mt1:mt2) == min_avg(nt,1)) - 1) <= reject_trial_before)
            %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26),'r'); hold on;
            bad_move_trials = [bad_move_trials; nt];
        else
            %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26)); hold on;
            good_move_trials = [good_move_trials; nt];
        end
    end

    peak_times = move_erp_time(mt1 + min_avg(:,2) - 1);
    [sorted_peak_times,sorting_order] = sort(peak_times,'descend');
    sorted_move_epochs_s = move_epochs_s(sorting_order,:,:);
    sorted_move_ch_avg_ini = move_ch_avg_ini(sorting_order,:);
    
%% Plot ERP image plot for classifier channels after sorting
if plot_erp_image == 1
%     modified_EEG_data = EEG.data(:,:,sorting_order);
%     EEG.data = modified_EEG_data;
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
%     eeglab redraw;
    
% Sort Channels
unsorted_labels = [];
for j = 1:length(classchannels)
    unsorted_labels = [unsorted_labels '_' EEG.chanlocs(classchannels(j)).labels];
end
disp(['Unsorted order:', unsorted_labels]);
sorted_order = input('Enter sorting order [1 2 3 4]:'); 
classchannels = classchannels(sorted_order);


% ERP image plot for classchannels and spatial average
figure('Position',[1050 1300 3.5*116 2.5*116]); 
T_plot = tight_subplot(1,length(classchannels)+1,[0.01 0.01],[0.15 0.15],[0.1 0.1]);
average_width = 2;  % Vertical moveing average filter; takes 2 trials above and below the trial of interest
avergae_type = 'boxcar'; % Avoid gaussian because avewidth of 1, applies a vertical window of 7 trials
decimate = [];
caxis_range = [-10 10];

for ch = 1:length(classchannels)
    axes(T_plot(ch));
    erpdata = squeeze(sorted_move_epochs_s(:,:,classchannels(ch)));
    [outdata,outvar,outtrials,limits,axhndls, ...
                        erp,amps,cohers,cohsig,ampsig,outamps,...
                        phsangls,phsamp,sortidx,erpsig] = erpimage(erpdata',[],move_erp_time,EEG.chanlocs(classchannels(ch)).labels,average_width,decimate,...
                                                                                                            'limits',[-2.5 1],'caxis',caxis_range,...%'vert',[-2 0.5],'horz',[132],...
                                                                                                            'cbar','off','cbar_title','\muV',...
                                                                                                            'noxlabel','on','avg_type',avergae_type,...
                                                                                                            'img_trialax_label',[],'img_trialax_ticks',[50 100 150]);
                                                                                                        
    img_handle = axhndls(1);
    img_xlab_handle = get(img_handle,'xlabel');
    img_ylab_handle = get(img_handle,'ylabel');
    img_title_handle = get(img_handle,'title');
    set(img_title_handle,'FontSize',paper_font_size-1,'FontWeight','normal');
    set(img_handle,'YLim',[1 size(move_epochs_s,1)],'Box','on')
    set(img_handle,'YTickLabel','');
    set(img_handle,'TickLength',[0.03 0.025]);
    set(img_handle,'Xtick',[-2 0 1]);
    set(img_handle,'XTickLabel',{' '},'FontSize',paper_font_size-1);
    
    if ch == 1
        set(img_xlab_handle,'String','Time (s)','FontSize',paper_font_size-1);
        set(img_ylab_handle,'String','Sorted Single Trials','FontSize',paper_font_size-1);  
        pos_ylab = get(img_ylab_handle,'Position');
        set(img_ylab_handle,'Position',[pos_ylab(1) - 0.8 pos_ylab(2) pos_ylab(3)]);
        set(img_handle,'Ytick',[50 100 150],'YTickLabel',{'50' '100' '150'},'FontSize',paper_font_size-1);
        set(img_handle,'XTickLabel',{'-2' 'MO' '1'},'FontSize',paper_font_size-1);
    end
    img_handle_children = get(img_handle,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
    set(img_handle_children(1),'LineWidth',0.5,'LineStyle','--');
end

     axes(T_plot(length(classchannels) + 1));
    [outdata,outvar,outtrials,limits,axhndls, ...
                    erp,amps,cohers,cohsig,ampsig,outamps,...
                    phsangls,phsamp,sortidx,erpsig] = erpimage(sorted_move_ch_avg_ini',[],move_erp_time,{'Spatial'; 'Average'},average_width,decimate,...
                                                                                                        'limits',[-2.5 1],'caxis',caxis_range,'vert',[-1.5],'horz',length(good_move_trials)+1,...
                                                                                                        'cbar','on','cbar_title','\muV',...
                                                                                                        'noxlabel','on','avg_type',avergae_type,...
                                                                                                        'img_trialax_label',[],'img_trialax_ticks',[50 100 150]);

    img_handle = axhndls(1);
    img_title_handle = get(img_handle,'title');
    set(img_title_handle,'FontSize',paper_font_size-1,'FontWeight','normal');
    set(img_handle,'TickLength',[0.03 0.025],'Box','on');
     set(img_handle,'YTickLabel','');
     set(img_handle,'Xtick',[-1.5]);
    set(img_handle,'XTickLabel',{'-1.5s'},'FontSize',paper_font_size-1);
    img_handle_children = get(img_handle,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
    set(img_handle_children(1),'LineWidth',0.5,'LineStyle','--');
    set(img_handle_children(2),'LineWidth',1,'Color',[0 0 0]);
    set(img_handle_children(3),'LineWidth',1,'Color',[0 0 0],'LineStyle','-');
    
    
    cbar_handle = axhndls(2);
    cbar_child_handle = get(cbar_handle,'child');
    cbar_title_handle = get(cbar_handle,'title');
    set(cbar_title_handle,'FontSize',paper_font_size-2);
    bar_pos = get(cbar_handle,'Position');
    set(cbar_handle,'Position',[bar_pos(1)+0.01 bar_pos(2) bar_pos(3)/3 bar_pos(4)/3]);
    ytickpos = get(cbar_handle,'Ytick');
    set(cbar_handle,'Ytick',[ytickpos(1) ytickpos(3) ytickpos(end)]);
    set(cbar_handle,'YtickLabel',[-10 0 10],'FontSize',paper_font_size-2);
    set(cbar_handle,'YDir','reverse');

% % Added 3-3-15
% cbar_axes = colorbar('location','SouthOutside','XTick',[-6 0 6],'XTickLabel',{'-6','0','6'});
% set(cbar_axes,'Position',[0.75 0.28 0.2 0.05]);
% xlabel(cbar_axes,'Average MRCP signal strength (\muV)','FontSize',18);

% Expand axes to fill figure
fig = gcf;
style = hgexport('factorystyle');
style.Bounds = 'tight';
hgexport(fig,'-clipboard',style,'applystyle', true);
drawnow;

response = input('Save figure to folder [y/n]: ','s');
if strcmp(response,'y')
     tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_erpimage.tif'];
     fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_erpimage.fig'];
    %print('-dtiff', '-r300', tiff_filename); 
    %saveas(gcf,fig_filename);
else
    disp('Save figure aborted');
end
end
%% Plot electrode locations
if plot_electrode_locations == 1
    figure('Position',[50 300 6*116 6*116]);
    [h_topo] = ...
    topoplot_nikunj([],'C:\NRI_BMI_Mahi_Project_files\EEGLAB_13_1_1b\BMI_Mahi_64Ch.locs','style','blank',...
                    'electrodes','ptslabels','plotchans',grand_average_chans,'plotrad', 0.6,...
                    'drawaxis', 'off', 'whitebk', 'off',...
                    'conv','off','emarker',{'o','k',24,2},'nosedir','+X','mrcp_chans',MRCP_chans,'classifier_chans',classchannels); 
    hall35 = get(get(gcf,'Children'),'Children');
    set(hall35(34),'Visible','off');
    set(hall35(35),'Visible','off');
    
    tightInset = get(gca, 'TightInset');        %Remove empty margin
    position(1) = tightInset(1);
    position(2) = tightInset(2);
    position(3) = 1 - tightInset(1) - tightInset(3);
    position(4) = 1 - tightInset(2) - tightInset(4);
    set(gca, 'Position', position);
    
    % Expand axes to fill figure
    fig = gcf;
    style = hgexport('factorystyle');
    style.Bounds = 'tight';
    hgexport(fig,'-clipboard',style,'applystyle', true);
    drawnow;
    tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_topoplot.tif'];
     fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_topoplot.fig'];
    print('-dtiff', '-r300', tiff_filename); 
    saveas(gcf,fig_filename);
    
end
%% Plot Scalp maps at different times
if plot_scalp_map == 1
Channels_used = [4 5 6 9 10 13 14 15 19 20 24 25 26 32 ...
                 38 39 43 44 48 49 52 53 54 57 58];

emarker_chans = [];
for ch_sel_cnt = 1:length(Channels_sel)
    emarker_chans(ch_sel_cnt) = find(Channels_used == Channels_sel(ch_sel_cnt));
end

time_stamps = [-2, -1, -0.5, 0, 0.5];
figure('units','normalized','outerposition',[0 0 1 1]);

Scalp_plot = tight_subplot(1,length(time_stamps),0.01,[0.1 0.05],[0.05 0.05]);

for tpt = 1:length(time_stamps)
    ScalpData = move_avg_channels(:,find(raster_time == time_stamps(tpt),1,'first')); 
    axes(Scalp_plot(tpt)); %#ok<LAXES>
    topoplot(ScalpData, EEG.chanlocs,'maplimits', [-6 6],'style','both',...    
        'electrodes','off','plotchans',Channels_used,'plotrad', 0.55,...
        'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off',...
        'conv','off',...
        'emarker2',{emarker_chans,'.','k',16,2});
    title(sprintf('%.1f',time_stamps(tpt)),'FontSize',18);
    %axis([-0.55 0.55 -1.82 1.82]);
end
title(sprintf('%.1f sec.',time_stamps(tpt)),'FontSize',18);
%colorbar('Position',[0.96 0.35 0.015 0.35]);
cbar_axes = colorbar('location','SouthOutside','XTick',[-6 0 6],'XTickLabel',{'-6','0','6'});
set(cbar_axes,'Position',[0.75 0.28 0.2 0.05]);
xlabel(cbar_axes,'Average MRCP signal strength (\muV)','FontSize',18);
%export_fig MS_ses1_cond3_Scalp_maps '-png' '-transparent';
end
%% Plot feature space and ROC, AUC
if plot_feature_space == 1
    figure('Position',[50 300 6*116 5*116]); 
    T_plot = tight_subplot(2,3,[0.15 0.1],[0.15 0.1],[0.1 0.05]);
    
    scatter_set_sm = Performance.Smart_Features;
    no_smart_epochs = round(size(scatter_set_sm,1)/2);
    axes(T_plot(5));    
    % 1. 2D Scatter Plot - data_Set1
    features_names = {'Slope','-ve Peak', 'Area', 'Mahalanobis'};
    %features_to_plot = [2 3 4]; % [X Y Z]
    features_to_plot = [1 4]; % [X Y]
    %scolor =  [repmat([0 0 0],no_smart_epochs,1);repmat([0.6 0.6 0.6],no_smart_epochs,1)]; 
    groups = [ones(no_smart_epochs,1); 2*ones(no_smart_epochs,1)]; 
    %hscatter_sm = scatter3(scatter_set_sm(:,features_to_plot(1)),scatter_set_sm(:,features_to_plot(2)),scatter_set_sm(:,features_to_plot(3)),4,scolor,'filled');
    hscatter_sm = gscatter(scatter_set_sm(:,features_to_plot(1)),scatter_set_sm(:,features_to_plot(2)),groups,'kr','+^',...
                                                [],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});
    %set(gcf,'Renderer','zbuffer');
    %set(hscatter_sm,'SizeData',20);
    min_feature_val = min(scatter_set_sm);
    max_feature_val = max(scatter_set_sm);
    xlim([5*floor(min_feature_val(features_to_plot(1))/5) 5*ceil(max_feature_val(features_to_plot(1))/5)+1]);      % round to nearest(above for max; below for min) multiple of 5
    ylim([5*floor(min_feature_val(features_to_plot(2))/5) 5*ceil(max_feature_val(features_to_plot(2))/5)]);
    %zlim([min_feature_val(features_to_plot(3)) max_feature_val(features_to_plot(3))]);
    %xlabel(features_names{features_to_plot(1)},'FontSize',10);
    %ylabel(features_names{features_to_plot(2)},'FontSize',10);
    %zlabel(features_names{features_to_plot(3)},'FontSize',10);
    set(hscatter_sm(1),'LineWidth',1);
    hparent = get(hscatter_sm(1),'parent');
    %set(hparent,'XTick',[-20 -10 0 10 15],'XtickLabel',{'-20' '-10' '0' '10' '15'});
    set(hscatter_sm(2),'LineWidth',1);
    
     title('Adaptive window','FontSize',10);
%     set(hscatter_sm(1),'DisplayName','Go');
%     set(hscatter_sm(2),'DisplayName','No-go');
     
   
    scatter_set_conv = Performance_conv.Conventional_Features;
    no_conv_epochs = round(size(scatter_set_conv,1)/2);
    axes(T_plot(4));    
    % 1. Scatter Plot - data_Set1
    %scolor =  [repmat([0 0 0],no_conv_epochs,1);repmat([0.4 0.4 0.4],no_conv_epochs,1)]; 
    %hscatter_conv = scatter3(scatter_set_conv(:,features_to_plot(1)),scatter_set_conv(:,features_to_plot(2)),scatter_set_conv(:,features_to_plot(3)),4,scolor,'filled');
    %set(gcf,'Renderer','zbuffer');
    %set(hscatter_conv,'SizeData',20);
    
     groups = [ones(no_conv_epochs,1); 2*ones(no_conv_epochs,1)]; 
    hscatter_conv = gscatter(scatter_set_conv(:,features_to_plot(1)),scatter_set_conv(:,features_to_plot(2)),groups,'kr','+^',...
                                                [],'on',features_names{features_to_plot(1)}, features_names{features_to_plot(2)});
                                            
    xlim([5*floor(min_feature_val(features_to_plot(1))/5) 5*ceil(max_feature_val(features_to_plot(1))/5)+1]);      % round to nearest(above for max; below for min) multiple of 5
    ylim([5*floor(min_feature_val(features_to_plot(2))/5) 5*ceil(max_feature_val(features_to_plot(2))/5)]);
    set(hscatter_conv(1),'DisplayName','Go (wl_o)');
    set(hscatter_conv(2),'DisplayName','No-go (wl_o)');
    set(hscatter_conv(1),'LineWidth',1);
    set(hscatter_conv(2),'LineWidth',1);
    title('Fixed window','FontSize',10);
    leg_scatter= findobj(gcf,'tag','legend'); set(leg_scatter,'location','northeast');
    leg_scatter_pos = get(leg_scatter,'Position');
    %set(leg_scatter,'position',[mod_leg_auc_pos(1) mod_leg_auc_pos(2)-leg_scatter_pos(4) leg_scatter_pos(3) leg_scatter_pos(4)]);
    axes6pos = get(T_plot(6),'Position');
    set(leg_scatter,'position',[axes6pos(1)-0.05 axes6pos(2)+axes6pos(4)-leg_scatter_pos(4) leg_scatter_pos(3) leg_scatter_pos(4)]);

    set(leg_scatter,'Box','on','FontSize',8);

    % Plot ROC curves for fixed and adaptive window features
    all_window_sizes = Performance.All_window_length_range;
    for wl_ind = 1:length(all_window_sizes)
            roc_xy_conv = Performance_conv.roc_X_Y_Thr{wl_ind,1};
            roc_xy_smart = Performance.roc_X_Y_Thr{wl_ind,2}(:,:,2);
            axes(T_plot(1)); hold on;
            if all_window_sizes(wl_ind) == all_window_sizes(Performance_conv.conv_opt_wl_ind)
                %plot(roc_xy_conv(:,1),roc_xy_conv(:,2),'-k','LineWidth',1.5);
                %line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
            else
                plot(roc_xy_conv(:,1),roc_xy_conv(:,2),'-','Color',[0.7 0.7 0.7],'LineWidth',1);
            end

            axes(T_plot(2)); hold on;
            if all_window_sizes(wl_ind) == all_window_sizes(Performance.smart_opt_wl_ind)
                %plot(roc_xy_smart(:,1),roc_xy_smart(:,2),'-k','LineWidth',1.5);
                %line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
            else
                plot(roc_xy_smart(:,1),roc_xy_smart(:,2),'-','Color',[0.7 0.7 0.7],'LineWidth',1);
            end
    end
    axes(T_plot(1)); hold on;
    plot(Performance_conv.roc_X_Y_Thr{Performance_conv.conv_opt_wl_ind,1}(:,1),...
             Performance_conv.roc_X_Y_Thr{Performance_conv.conv_opt_wl_ind,1}(:,2),'-k','LineWidth',1.5);
    line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
    text(0.25,0.07,sprintf('AUC (wl_o) = %.2f', Performance_conv.roc_OPT_AUC(Performance_conv.conv_opt_wl_ind,3)),'FontSize',10);
    title('Fixed window','FontSize',10);
    
    axes(T_plot(2)); hold on;
    h_smart_roc =  plot(Performance.roc_X_Y_Thr{Performance.smart_opt_wl_ind,2}(:,1,2),...
                                          Performance.roc_X_Y_Thr{Performance.smart_opt_wl_ind,2}(:,2,2),'-k','LineWidth',1.5);
    line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
    text(0.25,0.07,sprintf('AUC (wl_o) = %.2f', Performance.roc_OPT_AUC(Performance.smart_opt_wl_ind,3,2)));
    title('Adaptive window','FontSize',10);
    
    xlabel(T_plot(1),'FPR','FontSize',10);
    ylabel(T_plot(1),'TPR','FontSize',10); 
    % title(T_plot(5),'Fixed Window','FontSize',10);
    % title(T_plot(5),'Adaptive Window','FontSize',10);
    xlabel(T_plot(2),'FPR','FontSize',10);
    ylabel(T_plot(2),'TPR','FontSize',10); 
    set(T_plot(1),'XTick',[0 0.5 1],'ytick',[0 0.5 1],'XtickLabel',{'0' '0.5' '1'},'YtickLabel',{'0' '0.5' '1'},'XGrid','on','YGrid','on','GridLineStyle',':');
    set(T_plot(2),'XTick',[0 0.5 1],'ytick',[0 0.5 1],'XtickLabel',{'0' '0.5' '1'},'YtickLabel',{'0' '0.5' '1'},'XGrid','on','YGrid','on','GridLineStyle',':');
    
    % Plot AUC for different window lengths and mark optimal window lengths
    axes(T_plot(3)); hold on;
    hconv_auc = plot(all_window_sizes./Fs_eeg, Performance_conv.roc_OPT_AUC(:,3),'-b','LineWidth',1);
    plot(all_window_sizes./Fs_eeg, Performance_conv.roc_OPT_AUC(:,3),'ob','LineWidth',1);
    hopt_auc = plot(Performance_conv.conv_window_length, Performance_conv.roc_OPT_AUC(Performance_conv.conv_opt_wl_ind,3),'ok','LineWidth',2);
    
    hsmart_auc = plot(all_window_sizes./Fs_eeg, Performance.roc_OPT_AUC(:,3,2),'-r','LineWidth',1);
    plot(all_window_sizes./Fs_eeg, Performance.roc_OPT_AUC(:,3,2),'or','LineWidth',1);
    plot(Performance.smart_window_length,Performance.roc_OPT_AUC(Performance.smart_opt_wl_ind,3,2),'ok','LineWidth',2);
    
    line([0 1.5],[0.5 0.5],'LineStyle','--', 'LineWidth',1.5,'Color','k');
    xlim([all_window_sizes(1)./Fs_eeg-0.05 all_window_sizes(end)./Fs_eeg+0.05]);
    ylim([0.3 0.9]);
    set(T_plot(3),'XTick',[0.5 0.6 0.7 0.8 0.9 1],'ytick',[0 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1],'XtickLabel',{'0.5' '0.6' '0.7' '0.8' '0.9' '1'},'YtickLabel',{'0' '0.2' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1'},'XGrid','on','YGrid','on','GridLineStyle',':');
    xlabel(T_plot(3),'Window lengths (sec.)','FontSize',10);
    ylabel(T_plot(3),'AUC','FontSize',10);
    set(T_plot(3),'Box','on');
    % Legends - no longer used
%    leg_auc = legend([hopt_auc h_smart_roc hconv_auc hsmart_auc],{['optimal window' char(10)  'length (wl_o)'],'ROC for wl_o', 'Fixed window', 'Adaptive window'});
%     leg_auc_pos = get(leg_auc,'Position');
%     axes6pos = get(T_plot(6),'Position');
%     set(leg_auc,'position',[axes6pos(1) - (leg_auc_pos(3) - axes6pos(3)) axes6pos(2)+axes6pos(4)-leg_auc_pos(4) leg_auc_pos(3) leg_auc_pos(4)]);
%     mod_leg_auc_pos = get(leg_auc,'Position');
%     leg_scatter_pos = get(leg_scatter,'Position');
%     set(leg_scatter,'position',[mod_leg_auc_pos(1) mod_leg_auc_pos(2)-leg_scatter_pos(4) leg_scatter_pos(3) leg_scatter_pos(4)]);
%     set(leg_auc,'Box','off');
%     set(leg_scatter,'Box','off');
    set(T_plot(6),'Visible','off');
    
    % Expand axes to fill figure
    fig = gcf;
    style = hgexport('factorystyle');
    style.Bounds = 'tight';
    hgexport(fig,'-clipboard',style,'applystyle', true);
    drawnow;

    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
         tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_feature_space_roc.tif'];
         fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_feature_space_roc.fig'];
        print('-dtiff', '-r300', tiff_filename); 
        saveas(gcf,fig_filename);
    else
        disp('Save figure aborted');
    end
end
%% Plot sensitivity & specificity for all subjects
if plot_offline_performance == 1
     figure('Position',[50 300 7*116 4.5*116]); 
    T_plot = tight_subplot(2,3,[0.05 0.02],[0.01 0.15],[0.1 0.05]);
    Subject_names = {'LSGR','PLSH','PLSH','ERWS','BNBO','BNBO'};
    Cond_nos = [3 3 1 3 3 1];
    
    
    for subj_n = 1:length(Subject_names)
                    Sess_num = '2';
                    Block_num = 160;
                    cond_n = Cond_nos(subj_n);
                    if subj_n == 1
                        Sess_num = '2b';
                        Block_num = 140;
                    end
                        
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(Sess_num) '\'];
                    Performance = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
                    chance_level_smart = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_chance_level_smart2.mat']);
                    Performance_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);
                    chance_level_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_chance_level_conventional2.mat']);

                    axes(T_plot(subj_n)); hold on;
                    %group_names = {{'Fixed';'Adaptive'},{'Window';'Window'}};
                    offline_conv = Performance_conv.All_eeg_accur{Performance_conv.conv_opt_wl_ind}(1,:);
                    offline_smart = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
                    h_offline_acc = boxplot([offline_conv' offline_smart'],'positions', [1 2.5],'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
                    set(h_offline_acc,'LineWidth',1);
                    set(h_offline_acc(5,1),'Color','k'); set(h_offline_acc(6,1),'Color','k');
                    set(h_offline_acc(5,2),'Color','k'); set(h_offline_acc(6,2),'Color','k');
                    if size(h_offline_acc,1) > 6
                        set(h_offline_acc(7,1),'MarkerEdgeColor','k','MarkerSize',4); 
                        set(h_offline_acc(7,2),'MarkerEdgeColor','k','MarkerSize',4);       % Color of outliers
                    end
                                                           
                    ylim([20 100]);
                    %xlim([1 5]);
                    set(gca,'YTick',[20:10:100]);
                    set(gca,'YTickLabel','');
                    if (subj_n == 1) || (subj_n == 4)
                        set(gca,'YTick',[20:10:100]);
                        set(gca,'YTickLabel',{'' '30' '' '50' '' '70' '' '90' ''});
                        %ylabel('Classification Accuracy (%)','FontSize',myfont_size,'FontWeight',myfont_weight);
                        ylabel('Offline Accuracy (%)','FontSize',10);
                    end
                    set(gca,'XtickLabel',{' '})
                    
                    switch subj_n
                        case 1
                            title_str = 'S1 (AT)';
                        case 2
                            title_str = 'S2 (AT)';
                        case 3
                            title_str = 'S2 (\bfBD\rm)';
                        case 4
                            title_str = 'S3 (AT)';
                        case 5
                            title_str = 'S4 (AT)';
                        case 6
                            title_str = 'S4 (\bfBD\rm)';
                    end
                    title(title_str,'FontSize',10);
                    %set(T_plot(subj_n),'YGrid','on','GridLineStyle',':');
                    text(0.9,25,sprintf('(%.2fs)',Performance_conv.conv_window_length),'FontSize',10);
                    text(2.45,25,sprintf('(%.2fs)',Performance.smart_window_length),'FontSize',10);

                    chance_test1 = chance_level_conv.Chance_accuracy_permutations_labels;
                    chance_test2 = chance_level_smart.Chance_accuracy_permutations_labels;
                     h_chance = boxplot(T_plot(subj_n), [chance_test1 chance_test2],...
                                                         'positions', [1.6 3.1],'plotstyle','traditional','widths',0.5,'boxstyle','outline','colors','k','symbol','o'); % symbol - Outliers take same color as box
                   
                     if size(h_offline_acc,1) > 6
                        set(h_chance(7,1),'MarkerEdgeColor','k','MarkerSize',4); 
                        set(h_chance(7,2),'MarkerEdgeColor','k','MarkerSize',4);       % Color of outliers
                    end
                    
                    set(h_chance,'LineWidth',1);
                    ylim([20 100]); xlim([0.5 3.5]);
                    set(gca,'Xtick',[1.3 2.8]);
                    set(gca,'XtickLabel',{' '})
%                     if (subj_n == 1) || (subj_n == 2) || (subj_n == 3)
%                         set(gca,'XTickLabel',{' '});                                                                                                         % Hide group label names
%                     end
                    
                    h_colors = findobj(gca,'Tag','Box');
                    box_colors = [0.6 0.6 0.6; 0.6 0.6 0.6; 1 0 0; 0 0 1]; %reverse order
                    for j = 1:length(h_colors)
                        h_patch(j) = patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),box_colors(j,:));
                    end 
                     if subj_n == 2
                         axes_pos = get(gca,'Position');    % [left bottom width height]
                         box_leg = legend([h_patch(4) h_patch(3) h_patch(2)],'Fixed window', 'Adaptive window', 'Chance level', 'location','NorthOutside','Orientation','horizontal');
                         box_leg_pos = get(box_leg,'position');
                         box_leg_title = get(box_leg,'title');
                         set(box_leg_title,'String','Classifier performance during offline calibration');
                         set(box_leg,'position',[box_leg_pos(1), box_leg_pos(2)+0.06, box_leg_pos(3:4)]);
                     end
                     
                    % Significance tests
                    p_values = [];
                    if (chance_level_conv.p_labels_m2 <= 0.05) && (chance_level_conv.p_labels_m2 > 0.01) 
                        p_values = [p_values 0.05];
                    elseif (chance_level_conv.p_labels_m2 <= 0.01)
                        p_values = [p_values 0.01];
                    else
                        p_values = [p_values NaN];
                    end
                    
                    if (chance_level_smart.p_labels_m2 <= 0.05) && (chance_level_smart.p_labels_m2 > 0.01) 
                        p_values = [p_values 0.05];
                    elseif (chance_level_smart.p_labels_m2 <= 0.01)
                        p_values = [p_values 0.01];
                    else
                        p_values = [p_values NaN];
                    end
                    
                    [pwilcoxon,h,stats] = ranksum(offline_smart,offline_conv,'alpha',0.05,'tail','right');
                    if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                        p_values = [p_values 0.05];
                    elseif (pwilcoxon <= 0.01)
                        p_values = [p_values 0.01];
                    else
                        p_values = [p_values NaN];
                    end
                    
                    sigstar({[1 1.6],[2.5 3.1],[1 2.5]},p_values);
                    
    end

%     fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
%     %hgexport(fig,'-clipboard',style,'applystyle', true);
%     drawnow;

    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
        tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.tif'];
        fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.fig'];
        print('-dtiff', '-r300', tiff_filename); 
        saveas(gcf,fig_filename);
        im = imread(tiff_filename, 'tiff' );
        [im_hatch,colorlist] = applyhatch_pluscolor(im,'\/',0,0,[0 0 1; 1 0 0],300,3,2);
        imwrite(im_hatch,tiff_filename,'tiff');        
    else
        disp('Save figure aborted');
    end
end
%% Plot EEG-EMG latency
if plot_eeg_emg_latencies == 1

figure('Position',[50 200 9*116 6*116])    
Subject_names = {'LSGR','PLSH','ERWS','BNBO'};
Sess_nums = 4:5;
edges = [0 0.01 0.1 0.2 0.4 0.6 0.8 1];

for subj_n = 1:4
                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_closeloop_eeg_emg_latencies.mat']);      % Always use causal for training classifier
                    
                    Nfreq = histc(EEG_EMG_latency_calculated,edges);
                    plot_num = 2*subj_n; 
                    if n == 1
                        subplot(4,2,plot_num - 1);
                    elseif n == 2
                        subplot(4,2,plot_num);
                    end
                    Nfreq_norm = Nfreq/sum(Nfreq);
                    hbar = bar(edges,Nfreq_norm,'histc');
                    delete(findobj('marker','*'));
                    set(hbar,'LineWidth',1,'FaceColor',[1 0 0],'Facealpha',0.8);
                    hbar_parent = get(hbar,'parent');
                    set(hbar_parent,'XLim',[0 1],'XTick',[0.01 0.1 0.2 0.4 0.6 0.8 1],'XtickLabel',{'.01' '0.1' '0.2' '0.4' '0.6' '0.8' '1'});
                    if subj_n == 4
                        xlabel('Latency (sec)');
                    end
                    set(hbar_parent,'YLim',[0 max(Nfreq_norm) + 0.2]);
                    if n == 1
                        ylabel({'Normalized';'Frequency'});
                    end
                    title(['S' num2str(subj_n) ',triggered']);
                    if subj_n == 2 || subj_n == 4
                        if ses_n == 4
                            h_title = title(['S' num2str(subj_n) ',backdrive']);
                            set(h_title,'Color',[1 0 0]);
                            set(h_title,'FontWeight','bold');
                        end
                    end
                            
                    
                end
end
mtit('EEG-EMG intent detection latency','fontsize',10,'color',[0 0 0],'xoff',0,'yoff',.025);                    
% Expand axes to fill figure
fig = gcf;
style = hgexport('factorystyle');
style.Bounds = 'tight';
hgexport(fig,'-clipboard',style,'applystyle', true);
drawnow;

response = input('Save figure to folder [y/n]: ','s');
if strcmp(response,'y')
     tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subject_eeg_emg_latency.tif'];
     fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subject_eeg_emg_latency'];
    print('-dtiff', '-r300', tiff_filename); 
    saveas(gcf,fig_filename);
else
    disp('Save figure aborted');
end

end
%% Plot Single Trial Spatial Averages 
if plot_single_spatial_average_trials == 1
   figure('Position',[1050 1100 3.5*116 2*116]); 
   T_plot = tight_subplot(1,2,[0.02 0.02],[0.15 0.1],[0.15 0.01]);
   
   
   [unique_peak_times,peak_times_index,~] = unique(sorted_peak_times);
   %index_trials_to_plot = downsample(peak_times_index,8); % 10 trials
   index_trials_to_plot = [20 67 100 126 145];
   trial_time = move_erp_time(find(move_erp_time == -2.5):find(move_erp_time == 1));
   spatial_single_trials = sorted_move_ch_avg_ini(index_trials_to_plot,find(move_erp_time == -2.5):find(move_erp_time == 1));
   normalized_spatial_single_trials = zscore(spatial_single_trials');
   %normalized_spatial_single_trials = spatial_single_trials;
   %
   axes(T_plot(1)); hold on;
%    for i = 1:length(index_trials_to_plot)
%        if sorted_peak_times(index_trials_to_plot(i)) <= reject_trial_before
%            % plot dotted black
%            h1 = plot(trial_time,normalized_spatial_single_trials(i,:),'LineStyle','--','LineWidth',1,'Color',[0.6 0.6 0.6]);           
%        else
%            % plot solid black
%            h2 = plot(trial_time,normalized_spatial_single_trials(i,:),'LineStyle','-','LineWidth',1,'Color',[0.6 0.6 0.6]);
%        end
%    end
   raster_zscore = 0.8.*normalized_spatial_single_trials;
    raster_time = trial_time;
    %raster_colors = ['m','k','b','r','k','b','r'];
    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 4;
    %raster_ylim1 = 0;
    %raster_ylim2 = (raster_col+1)*add_offset;
%      raster_zscore(:,1:3) = raster_zscore(:,1:3).*0.5;
%     raster_zscore(:,5) = raster_zscore(:,5).*0.5;
%     raster_zscore(:,4) = raster_zscore(:,4).*0.5;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*(raster_index);  % Add offset to each channel of raster plot
        %myhand(raster_index) = plot(raster_time,raster_zscore(:,raster_index),'Color',[0 0 0]);
        %jbfill(trial_time',repmat(max(raster_zscore(:,raster_index)),1,length(trial_time)),raster_zscore(:,raster_index)',[0.6 0.6 0.6],'k',0,0.3);
        jbfill(trial_time',repmat(max(raster_zscore(:,raster_index)),1,length(trial_time)),raster_zscore(:,raster_index)',[0.5 0.5 0.5],[0.5 0.5 0.5],0,0.5);
        hold on;
    end
    
    set(gca,'Xtick',[-2 -1 0 1],'XtickLabel',{'-2' '-1' 'MO' '1'},'xgrid','on','box','on');
     axis([-2.5 1 min(raster_zscore(:,1)) max(raster_zscore(:,end))]);
   %text(-0.3,4.5,'detected');
   set(gca,'Ydir','reverse');
   %ylim([20 50]);
    
    hxlab = xlabel('Time (sec.)'); 
    %pos_hxlab = get(hxlab,'Position');
    %set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) + 0.6) pos_hxlab(3)]);
    %set(gca,'Ytick',[-10 -5 0 5 10],'YtickLabel',{'-10' '-5' '0' '5' '10'},'ygrid','on','box','on');
    hylab = ylabel({'Standardized Amplitude'});
    pos_hylab = get(hylab,'Position');
    set(hylab,'Position',[(pos_hylab(1) - 0.5) pos_hylab(2) pos_hylab(3)]);
    %plot_leg = legend([h1 h2],'Mean (Offline training trials)', 'Single-trials (Online BMI Control)','location','NorthWest','Orientation','vertical');
    %set(plot_leg,'FontSize',8,'box','off');
    %pos_leg = get(plot_leg,'Position');
    %axpos = get(gca,'Position');
    %set(plot_leg,'Position',[axpos(1)-0.005 (axpos(2)+axpos(4) - pos_leg(4)) pos_leg(3) pos_leg(4)]);
    
    title({'Single-trials'});
    ylimits = ylim;
    %line([0 0],ylimits,'Color','k','LineWidth',0.5,'LineStyle','--');   
    shd1 = find(trial_time == (0.5-Performance_conv.conv_window_length),1,'first');
    shd2 = find(trial_time == 0.5,1,'first');
    h_patch(1) = jbfill(trial_time(shd1:shd2)',repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'b','b',0,0.3);

    % Annotate line
    axes_pos = get(gca,'Position'); %[lower bottom width height]
    axes_ylim = get(gca,'Ylim');
    annotate_length = (5*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
    annotation(gcf,'line', [(axes_pos(1) - 0.025) (axes_pos(1) - 0.025)],...
        [(axes_pos(2)+axes_pos(4) - annotate_length/5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/5)],'LineWidth',0.5);
    
    %%%----------------------------------------------------------------------------
    axes(T_plot(2)); hold on;
%    for i = 1:length(index_trials_to_plot)
%        if sorted_peak_times(index_trials_to_plot(i)) <= reject_trial_before
%            % plot dotted black
%            h1 = plot(trial_time,normalized_spatial_single_trials(i,:),'LineStyle','--','LineWidth',1,'Color',[0.6 0.6 0.6]);           
%        else
%            % plot solid black
%            h2 = plot(trial_time,normalized_spatial_single_trials(i,:),'LineStyle','-','LineWidth',1,'Color',[0.6 0.6 0.6]);
%        end
%    end
    raster_zscore = 0.8.*normalized_spatial_single_trials;
    raster_time = trial_time;
    %raster_colors = ['m','k','b','r','k','b','r'];
    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 4;
    %raster_ylim1 = 0;
    %raster_ylim2 = (raster_col+1)*add_offset;
%      raster_zscore(:,1:3) = raster_zscore(:,1:3).*0.5;
%     raster_zscore(:,5) = raster_zscore(:,5).*0.5;
%     raster_zscore(:,4) = raster_zscore(:,4).*0.5;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
        %myhand(raster_index) = plot(raster_time,raster_zscore(:,raster_index),'Color',[0 0 0]);
        jbfill(trial_time',repmat(max(raster_zscore(:,raster_index)),1,length(trial_time)),raster_zscore(:,raster_index)',[0.5 0.5 0.5],[0.5 0.5 0.5],0,0.5);
        
        start_time = round((sorted_peak_times(index_trials_to_plot(raster_index))-Performance.smart_window_length)*100)/100;
        red_shd1 = find(trial_time == start_time,1,'first');
        red_shd2 = find(trial_time == (sorted_peak_times(index_trials_to_plot(raster_index))),1,'first');
        if ~isempty(red_shd1)
            h_patch(2) = jbfill(trial_time(red_shd1:red_shd2)',...
                      repmat(max(raster_zscore(:,raster_index)),1,red_shd2-red_shd1+1),...
                      raster_zscore(red_shd1:red_shd2,raster_index)','r','r',0,0.5);
        else
            h_cross = plot(sorted_peak_times(index_trials_to_plot(raster_index)), raster_zscore(red_shd2,raster_index),'xr','MarkerSize',8);
            %line([sorted_peak_times(index_trials_to_plot(raster_index)) sorted_peak_times(index_trials_to_plot(raster_index))],...
             %       [raster_zscore(red_shd2,raster_index) max(raster_zscore(:,raster_index))],'Color','r','LineWidth',0.5,'LineStyle','--');
        end
        hold on;
    end
    ylimits = ylim;
    %line([0 0],ylimits,'Color','k','LineWidth',0.5,'LineStyle','--');
    h_criteria = line([-1.5 -1.5],ylimits,'Color','k','LineWidth',0.5,'LineStyle','-');
    
    set(gca,'Xtick',[-2 -1 0 1],'XtickLabel',{'-2' '-1' 'MO' '1'},'xgrid','on','box','on');
    axis([-2.5 1 min(raster_zscore(:,1)) max(raster_zscore(:,end))]);
   %text(-0.3,4.5,'detected');
   set(gca,'Ydir','reverse');
   %ylim([20 50]);
    
    hxlab = xlabel('Time (sec.)'); 
    title({'Single-trials'});
    
%  % Legends
%  leg_shading = legend([h_patch(1) h_patch(2) h_cross h_criteria],'Fixed window','Adaptive window', 'Rejected trial', 'Rejection rule');
%  leg_shad_pos = get(leg_shading,'Position');
%  axes3pos = get(T_plot(3),'Position');
%  set(leg_shading,'position',[axes3pos(1)-0.05 axes3pos(2)+axes3pos(4)-leg_shad_pos(4) leg_shad_pos(3) leg_shad_pos(4)]);
%  set(leg_shading,'Box','on','FontSize',8);
%  set(T_plot(3),'Visible','off')
%  
%  % Expand axes to fill figure
% fig = gcf;
% style = hgexport('factorystyle');
% style.Bounds = 'tight';
% hgexport(fig,'-clipboard',style,'applystyle', true);
% drawnow;
% 
%  response = input('Save figure to folder [y/n]: ','s');
%     if strcmp(response,'y')
%         tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_fixed_vs_adap_windows.tif'];
%         fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_fixed_vs_adap_windows.fig'];
%         print('-dtiff', '-r600', tiff_filename); 
%         saveas(gcf,fig_filename);
%     else
%         disp('Save figure aborted');
%     end
    
end
%% Merge figures for classifier design
if plot_classifier_design_all == 1
    
   figure('Position',[1050 1100 7.16*116 5*116]); 
   N_plot = tight_subplot(3,4,[0.1 0.05],[0.1 0.05],[0.05 0.01]);
     
   ax5pos = get(N_plot(5),'Position'); 
   axes(N_plot(1));
   ax1pos = get(gca,'Position');
   diff = 1*((ax1pos(2) - ax5pos(2))/3);
   set(gca,'Position',[ax1pos(1) ax5pos(2)+diff-0.04 ax1pos(3) ax1pos(4) + (ax1pos(2) - ax5pos(2) - diff)]);
   set(N_plot(5),'Visible','off'); 
   
    ax6pos = get(N_plot(6),'Position'); 
   axes(N_plot(2));
   ax2pos = get(gca,'Position');
   diff = 1*((ax2pos(2) - ax6pos(2))/3);
   set(gca,'Position',[ax2pos(1)-0.04 ax6pos(2)+diff-0.04 ax2pos(3) ax2pos(4)+(ax2pos(2) - ax6pos(2) - diff)]);
   set(N_plot(6),'Visible','off'); 
      
   axes(N_plot(3));
   ax3pos = get(gca,'Position'); 
   set(N_plot(3),'Position',[ax3pos(1)+0.04 (ax3pos(2)+ax3pos(4)/4)-0.04 ax3pos(3) (ax3pos(4)-ax3pos(4)/4)]);
   
   axes(N_plot(4));
   ax4pos = get(gca,'Position'); 
   set(gca,'Position',[ax4pos(1) (ax4pos(2)+ax4pos(4)/4)-0.04 ax4pos(3) (ax4pos(4)-ax4pos(4)/4)]);
   
   ax1pos_n = get(N_plot(1),'Position');
   axes(N_plot(7));
   ax7pos = get(gca,'Position'); 
   %set(gca,'Position',[ax7pos(1) (ax7pos(2)+ax7pos(4)/4-ax7pos(4)/8) ax7pos(3) (ax7pos(4)-ax7pos(4)/4)]);
   set(N_plot(7),'Position',[ax7pos(1)+0.04 ax1pos_n(2) ax7pos(3) (ax7pos(4)-ax7pos(4)/2)]);
   
   axes(N_plot(8));
   ax8pos = get(gca,'Position'); 
   %set(gca,'Position',[ax8pos(1) (ax8pos(2)+ax8pos(4)/4-ax8pos(4)/8) ax8pos(3) (ax8pos(4)-ax8pos(4)/4)]);
   set(N_plot(8),'Position',[ax8pos(1) ax1pos_n(2) ax8pos(3) (ax8pos(4)-ax8pos(4)/2)]);
   
   axes(N_plot(9));
   ax9pos = get(gca,'Position'); 
   set(N_plot(9),'Position',[ax9pos(1) ax9pos(2)+0.02 ax9pos(3) ax9pos(4)]);
   
   axes(N_plot(10));
   ax10pos = get(gca,'Position'); 
   set(N_plot(10),'Position',[ax10pos(1) ax10pos(2)+0.02 ax10pos(3) ax10pos(4)]);
   
   ax12pos = get(N_plot(12),'Position');
   ax7pos_n = get(N_plot(7),'Position');
   axes(N_plot(11));
   ax11pos = get(gca,'Position');
   set(gca,'Position',[ax11pos(1)+0.04 ax11pos(2)+0.02 ax11pos(3) + ((ax12pos(1)+ax12pos(3))-(ax11pos(1)+ax11pos(3)))-0.04 ax11pos(4)]);
   ax11pos_orig = get(N_plot(11),'Position');
   set(N_plot(12),'Visible','off'); 
   
%    axes(N_plot(10));
%    ax10pos = get(gca,'Position'); 
%    set(gca,'Position',[ax10pos(1) ax10pos(2:4)]);
   
   %% 1.  Plot single-trial amplitudes
   [unique_peak_times,peak_times_index,~] = unique(sorted_peak_times);
   %index_trials_to_plot = downsample(peak_times_index,8); % 10 trials
   index_trials_to_plot = [20 67 100 126 145];
   trial_time = move_erp_time(find(move_erp_time == -2.5):find(move_erp_time == 1));
   spatial_single_trials = sorted_move_ch_avg_ini(index_trials_to_plot,find(move_erp_time == -2.5):find(move_erp_time == 1));
   normalized_spatial_single_trials = zscore(spatial_single_trials');
   %normalized_spatial_single_trials = spatial_single_trials;
   %
   axes(N_plot(1)); hold on;
   raster_zscore = 0.7.*normalized_spatial_single_trials;
    raster_time = trial_time;
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 3.5;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*(raster_index);  % Add offset to each channel of raster plot
        %plot(trial_time', raster_zscore(:,raster_index),'-k','LineWidth',1)
        jbfill(trial_time',repmat(max(raster_zscore(:,raster_index)),1,length(trial_time)),raster_zscore(:,raster_index)',[0.5 0.5 0.5],[0.5 0.5 0.5],0,0.5);
        hold on;
    end
    
   set(gca,'Xtick',[-2 -1 0 1],'XtickLabel',{'-2' '-1' 'MO' '1'},'xgrid','on','box','on','FontSize',paper_font_size-1);
   axis([-2.5 1 min(raster_zscore(:,1)) max(raster_zscore(:,end))]);
   set(gca,'Ydir','reverse');   
    hxlab = xlabel('Time (sec.)','FontSize',paper_font_size-1); 
    %pos_hxlab = get(hxlab,'Position');
    %set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) + 0.6) pos_hxlab(3)]);
    %set(gca,'Ytick',[-10 -5 0 5 10],'YtickLabel',{'-10' '-5' '0' '5' '10'},'ygrid','on','box','on');
    hylab_shd = ylabel({'Standardized Amplitude'},'FontSize',paper_font_size-1);
    pos_hylab_shd = get(hylab_shd,'Position');
    set(hylab_shd,'Position',[pos_hylab_shd(1) pos_hylab_shd(2)+1 pos_hylab_shd(3)]);       
    title({'Single-trial Epochs'});
    ylimits = ylim;
    %line([0 0],ylimits,'Color','k','LineWidth',0.5,'LineStyle','--');   
    shd1 = find(trial_time == (0.5-Performance_conv.conv_window_length),1,'first');
    shd2 = find(trial_time == 0.5,1,'first');
    h_patch(1) = jbfill(trial_time(shd1:shd2)',repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'b','b',0,0.3);

    % Annotate line
%     axes_pos = get(gca,'Position'); %[lower bottom width height]
%     axes_ylim = get(gca,'Ylim');
%     annotate_length = (3.5*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
%     annotation(gcf,'line', [(axes_pos(1) - 0.02) (axes_pos(1) - 0.02)],...
%         [(axes_pos(2)+axes_pos(4) - annotate_length/3.5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/3.5)],'LineWidth',1);
    
    %%%----------------------------------------------------------------------------
    axes(N_plot(2)); hold on;
    raster_zscore = 0.7.*normalized_spatial_single_trials;
    raster_time = trial_time;
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 3.5;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
        jbfill(trial_time',repmat(max(raster_zscore(:,raster_index)),1,length(trial_time)),raster_zscore(:,raster_index)',[0.5 0.5 0.5],[0.5 0.5 0.5],0,0.5);
        
        start_time = round((sorted_peak_times(index_trials_to_plot(raster_index))-Performance.smart_window_length)*100)/100;
        red_shd1 = find(trial_time == start_time,1,'first');
        red_shd2 = find(trial_time == (sorted_peak_times(index_trials_to_plot(raster_index))),1,'first');
        
        if ~isempty(red_shd1)
            h_patch(2) = jbfill(trial_time(red_shd1:red_shd2)',...
                      repmat(max(raster_zscore(:,raster_index)),1,red_shd2-red_shd1+1),...
                      raster_zscore(red_shd1:red_shd2,raster_index)','r','r',0,0.5);
              plot(N_plot(2),trial_time(red_shd2),min(raster_zscore(:,raster_index))-0.2,'vr','MarkerFaceColor','w','MarkerSize',4)
        else
            %h_cross = plot(sorted_peak_times(index_trials_to_plot(raster_index)), raster_zscore(red_shd2,raster_index),'xr','MarkerSize',8);
            h_peak = plot(N_plot(2),trial_time(red_shd2),min(raster_zscore(:,raster_index))-0.13,'vr','MarkerFaceColor','w','MarkerSize',4)
        end
        hold on;
    end
    ylimits = ylim;
    %line([0 0],ylimits,'Color','k','LineWidth',0.5,'LineStyle','--');
    h_criteria = line([-1.5 -1.5],ylimits,'Color','k','LineWidth',0.5,'LineStyle','-');
    
    set(gca,'Xtick',[-2 -1.5 -1 0 1],'XtickLabel',{'-2' '-1.5' '-1' 'MO' '1'},'xgrid','on','box','on','FontSize',paper_font_size-1);
    axis([-2.5 1 min(raster_zscore(:,1))-0.5 max(raster_zscore(:,end))]);
   set(gca,'Ydir','reverse');
       
    hxlab = xlabel('Time (sec.)','FontSize',paper_font_size-1); 
    title({'Single-trial epochs'});    
    
    
    [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([h_patch(1), h_patch(2)],{'Fixed window','Adaptive window'},'ncol',2, 'ref',N_plot(1),...
                                            'anchor',[1 7],'buffer',[0 25],'box','off','xscale',0.3,'padding',[12 10 20]);
    set(object_h(3),'FaceAlpha',0.5);
    set(object_h(4),'FaceAlpha',0.5);
   
    
    [legend_h1,object_h1,plot_h1,text_str1] = ...
                        legendflex(h_peak,{'-ve Peak'},'ncol',1, 'ref',N_plot(2),...
                                            'anchor',[1 1],'buffer',[0 0],'box','off','xscale',0.3,'padding',[0 5 0]);
    
    
    %% 2. Plot feature space
    round_mul = 4;
    scatter_set_sm = Performance.Smart_Features;
    no_smart_epochs = round(size(scatter_set_sm,1)/2);
    scatter_set_conv = Performance_conv.Conventional_Features;
    no_conv_epochs = round(size(scatter_set_conv,1)/2);
    features_names = {'Slope','-ve Peak', 'Area', 'Mahalanobis'};
    % 1. 2D Scatter Plot - data_Set1
    features_to_plot = [1 4]; % [X Y]
    %scolor =  [repmat([0 0 0],no_smart_epochs,1);repmat([0.6 0.6 0.6],no_smart_epochs,1)]; 
    groups_conv = [ones(no_conv_epochs,1); 2*ones(no_conv_epochs,1)]; 
    groups_smart = [ones(no_smart_epochs,1); 2*ones(no_smart_epochs,1)]; 
    
    axes(N_plot(3));    
    hscatter_sm2 = gscatter(scatter_set_conv(:,features_to_plot(1)),scatter_set_conv(:,features_to_plot(2)),groups_conv,[[0 0 0];[0.7 0.7 0.7]],'xo',...
                                                [],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});
    min_feature_val = min(scatter_set_sm);
    max_feature_val = max(scatter_set_sm);
    %xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(1))/round_mul)+2]);      % round to nearest(above for max; below for min) multiple of round_mul
    xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) 10]);
    ylim([round_mul*floor(min_feature_val(features_to_plot(2))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(2))/round_mul)]);
    set(hscatter_sm2(1),'MarkerSize',6);set(hscatter_sm2(2),'MarkerSize',4);
    
    hparent = get(hscatter_sm2(1),'parent');
    set(hparent,'XTick',[-20 -10 0 10],'XtickLabel',{'-20' '-10' '0' '10'},'Ytick',[0 5 10],'YtickLabel',{'0' '5' '10'});  
    title('Fixed window','FontSize',paper_font_size-1);
    set(gca,'FontSize',paper_font_size-1);
%     set(hscatter_sm2(1),'DisplayName','Go (wl_o)');
%     set(hscatter_sm2(2),'DisplayName','No-go (wl_o)');
    
%     leg_scatter= findobj(gcf,'tag','legend'); 
%     set(leg_scatter,'location','northeast');
%     set(leg_scatter,'FontSize',8,'Box','off')
%     leg_scatter_pos = get(leg_scatter,'Position');
%     axes6pos = get(T_plot(6),'Position');
%     set(leg_scatter,'position',[axes6pos(1)-0.05 axes6pos(2)+axes6pos(4)-leg_scatter_pos(4) leg_scatter_pos(3) leg_scatter_pos(4)]);
%     set(leg_scatter,'Box','on','FontSize',8);
    
    axes(N_plot(4));    
    hscatter_sm1 = gscatter(scatter_set_sm(:,features_to_plot(1)),scatter_set_sm(:,features_to_plot(2)),groups_smart,[[0 0 0];[0.7 0.7 0.7]],'xo',...
                                                [],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});
    min_feature_val = min(scatter_set_sm);
    max_feature_val = max(scatter_set_sm);
    %xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(1))/round_mul)+1]);      % round to nearest(above for max; below for min) multiple of 5
    xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) 10]);
    ylim([round_mul*floor(min_feature_val(features_to_plot(2))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(2))/round_mul)]);
    set(hscatter_sm1(1),'MarkerSize',6);set(hscatter_sm1(2),'MarkerSize',4);
    hparent = get(hscatter_sm1(1),'parent');
    set(hparent,'XTick',[-20 -10 0 10],'XtickLabel',{'-20' '-10' '0' '10'},'Ytick',[0 5 10],'YtickLabel',{' '});  
    title('Adaptive window','FontSize',paper_font_size-1);
    ylabel(gca,' ');
    set(gca,'FontSize',paper_font_size-1);
    
    %Plot remaining features
    features_to_plot = [2 3]; % [X Y]
       
    axes(N_plot(7));    
    hscatter_sm3 = gscatter(scatter_set_conv(:,features_to_plot(1)),scatter_set_conv(:,features_to_plot(2)),groups_conv,[[0 0 0];[0.7 0.7 0.7]],'xo',...
                                                [],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});
    min_feature_val = min(scatter_set_sm);
    max_feature_val = max(scatter_set_sm);
    %xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(1))/round_mul)]);      % round to nearest(above for max; below for min) multiple of 5
    xlim([-15 5]);
    %ylim([round_mul*floor(min_feature_val(features_to_plot(2))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(2))/round_mul)]);
    ylim([-10 6]);
    set(hscatter_sm3(1),'MarkerSize',6);set(hscatter_sm3(2),'MarkerSize',4);
    hparent = get(hscatter_sm3(1),'parent');
    set(hparent,'XTick',[-10 -5  0 5],'XtickLabel',{'-10' '-5' '0' '5'},'Ytick',[-10 0 5],'YtickLabel',{'-10' '0' '5'});
    set(gca,'FontSize',paper_font_size-1);
       
    axes(N_plot(8));    
    hscatter_sm4 = gscatter(scatter_set_sm(:,features_to_plot(1)),scatter_set_sm(:,features_to_plot(2)),groups_smart,[[0 0 0];[0.7 0.7 0.7]],'xo',...
                                                [],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});
    min_feature_val = min(scatter_set_sm);
    max_feature_val = max(scatter_set_sm);
    %xlim([round_mul*floor(min_feature_val(features_to_plot(1))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(1))/round_mul)]);      % round to nearest(above for max; below for min) multiple of 5
    xlim([-15 5]);
    %ylim([round_mul*floor(min_feature_val(features_to_plot(2))/round_mul) round_mul*ceil(max_feature_val(features_to_plot(2))/round_mul)]);
    ylim([-10 6]);
    set(hscatter_sm4(1),'MarkerSize',6);set(hscatter_sm4(2),'MarkerSize',4);
    hparent = get(hscatter_sm4(1),'parent');
    set(hparent,'XTick',[-10 -5  0 5],'XtickLabel',{ '-10' '-5' '0' '5'},'Ytick',[-10 0 5],'YtickLabel',{' '});
    ylabel(gca,' ');
    set(gca,'FontSize',paper_font_size-1);
    
   legendflex([hscatter_sm4(1), hscatter_sm4(2)],{'Go','No-go'},'ncol',2, 'ref',N_plot(4),'anchor',[1 6],'buffer',[0 25],'box','off','xscale',0.3,'padding',[12 10 20]);
    %set(object_h(3),'FaceAlpha',0.5);
    %set(object_h(4),'FaceAlpha',0.5);
    %% 3. ROC curves
    all_window_sizes = Performance.All_window_length_range;
    for wl_ind = 1:length(all_window_sizes)
            roc_xy_conv = Performance_conv.roc_X_Y_Thr{wl_ind,1};
            roc_xy_smart = Performance.roc_X_Y_Thr{wl_ind,2}(:,:,2);
            axes(N_plot(9)); hold on;
            if all_window_sizes(wl_ind) == all_window_sizes(Performance_conv.conv_opt_wl_ind)
                %plot(roc_xy_conv(:,1),roc_xy_conv(:,2),'-k','LineWidth',1.5);
                %line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
            else
                plot(roc_xy_conv(:,1),roc_xy_conv(:,2),'-','Color',[0.5 0.5 1],'LineWidth',0.25);
            end

            axes(N_plot(9)); hold on;
            if all_window_sizes(wl_ind) == all_window_sizes(Performance.smart_opt_wl_ind)
                %plot(roc_xy_smart(:,1),roc_xy_smart(:,2),'-k','LineWidth',1.5);
                %line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
            else
                plot(roc_xy_smart(:,1),roc_xy_smart(:,2),'-','Color',[1 0.5 0.5],'LineWidth',0.25);
            end
    end
    axes(N_plot(9)); hold on;
    h_conv_roc = plot(Performance_conv.roc_X_Y_Thr{Performance_conv.conv_opt_wl_ind,1}(:,1),...
             Performance_conv.roc_X_Y_Thr{Performance_conv.conv_opt_wl_ind,1}(:,2),'-b','LineWidth',1.5);
    line([0 1],[0 1],'LineStyle','--', 'LineWidth',1,'Color','k');
    %text(0.25,0.07,sprintf('AUC (wl_o) = %.2f', Performance_conv.roc_OPT_AUC(Performance_conv.conv_opt_wl_ind,3)),'FontSize',10);
    %title('Fixed window','FontSize',10);
    
    axes(N_plot(9)); hold on;
    h_smart_roc =  plot(Performance.roc_X_Y_Thr{Performance.smart_opt_wl_ind,2}(:,1,2),...
                                          Performance.roc_X_Y_Thr{Performance.smart_opt_wl_ind,2}(:,2,2),'-r','LineWidth',1.5);
    %line([0 1],[0 1],'LineStyle','--', 'LineWidth',1.5,'Color','k');
    %text(0.25,0.07,sprintf('AUC (wl_o) = %.2f', Performance.roc_OPT_AUC(Performance.smart_opt_wl_ind,3,2)));
    title('ROC curves','FontSize',paper_font_size-1);
    
    xlabel(N_plot(9),'FPR','FontSize',paper_font_size-1);
    hylab_roc = ylabel(N_plot(9),'TPR','FontSize',paper_font_size-1); 
    pos_hylab_roc_orig = get(hylab_roc,'position');
    set(N_plot(9),'XTick',[0 0.5 1],'ytick',[0 0.5 1],'XtickLabel',{'0' ' ' '1'},'YtickLabel',{'0' ' ' '1'},'XGrid','on','YGrid','on');
    set(hylab_roc,'Position',[pos_hylab_roc_orig(1) pos_hylab_roc_orig(2) pos_hylab_roc_orig(3)]);
       
    % Plot AUC for different window lengths and mark optimal window lengths
    axes(N_plot(10)); hold on;
    hconv_auc = plot(all_window_sizes./Fs_eeg, Performance_conv.roc_OPT_AUC(:,3),'-b','LineWidth',1);
    plot(all_window_sizes./Fs_eeg, Performance_conv.roc_OPT_AUC(:,3),'.b','MarkerSize',8);
    hopt_auc = plot(Performance_conv.conv_window_length, Performance_conv.roc_OPT_AUC(Performance_conv.conv_opt_wl_ind,3),'ob','LineWidth',1);
    
    hsmart_auc = plot(all_window_sizes./Fs_eeg, Performance.roc_OPT_AUC(:,3,2),'-r','LineWidth',1);
    plot(all_window_sizes./Fs_eeg, Performance.roc_OPT_AUC(:,3,2),'.r','MarkerSize',8);
    plot(Performance.smart_window_length,Performance.roc_OPT_AUC(Performance.smart_opt_wl_ind,3,2),'or','LineWidth',1);
    
    line([0.5 1],[0.5 0.5],'LineStyle','--', 'LineWidth',1,'Color','k');
    xlim([all_window_sizes(1)./Fs_eeg all_window_sizes(end)./Fs_eeg]);
    ylim([0.4 1]);
    set(N_plot(10),'XTick',[0.5 0.75 1],'ytick',[0.5 0.6 0.8 1],'XtickLabel',{'0.5' '0.75' '1'},'YtickLabel',{'0.5' '0.6' '0.8' '1'},'XGrid','on','YGrid','on');
    xlabel(N_plot(10),'Window lengths (sec.)','FontSize',paper_font_size-1);
    title(N_plot(10),'Area under ROC','FontSize',paper_font_size-1);
    %set(N_plot(10),'Box','on');
   
    %% 4. Offline Performance
    axes(N_plot(11)); hold on;
    Subject_names = {'LSGR','PLSH','PLSH','ERWS','BNBO','BNBO'};
    Cond_nos = [3 3 1 3 3 1];
    group_conv_smart_perf = [];
    group_conv_smart_labels = [];
    sig_intervals = cell(1);
    p_values = [];
    
    for subj_n = 1:length(Subject_names)
                    Sess_num = '2';
                    Block_num = 160;
                    cond_n = Cond_nos(subj_n);
                    if subj_n == 1
                        Sess_num = '2b';
                        Block_num = 140;
                    end
                        
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(Sess_num) '\'];
                    Performance = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
                    Performance_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);                   
                    offline_conv = (Performance_conv.All_eeg_accur{Performance_conv.conv_opt_wl_ind}(1,:))';
                    offline_smart = (Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:))';
                    group_conv_smart_perf = [group_conv_smart_perf; [offline_conv;offline_smart]];
                    group_conv_smart_labels = [group_conv_smart_labels; [subj_n*ones(size(offline_conv,1),1); (subj_n+0.5)*ones(size(offline_smart,1),1)]];
                    
                     % Significance tests
                    [pwilcoxon,h,stats] = ranksum(offline_smart,offline_conv,'alpha',0.05,'tail','right');
                    if h == 1
                        sig_intervals(end+1) = {[subj_n-0.15 subj_n+0.15]};
                        if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                            p_values = [p_values 0.05];
                        elseif (pwilcoxon <= 0.01)
                            p_values = [p_values 0.01];
                        end
                     end
                    
    end
                    sig_intervals(1) = [];
                    group_names = {{'  S1';''; '  S2';''; '  S2';'';'  S3';'';'  S4';'';'  S4';''},{' (AT)';'';' (BD)';'';' (AT)';'';'  (AT)';'';'  (BD)';'';'  (AT)';''}};
                    group_positions = [0.85 1.15 1.85 2.15 2.85 3.15 3.85 4.15 4.85 5.15 5.85 6.15];
                    % Adaptive window
                    h_smart = boxplot(group_conv_smart_perf,group_conv_smart_labels,'labels', group_names, 'positions',group_positions,...
                                                        'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
                    set(h_smart,'LineWidth',1);
                    set(h_smart(5,1:2:12),'Color','b');         % Box
                    set(h_smart(5,2:2:12),'Color','r');
                    set(h_smart(6,1:2:12),'Color','b');          % Median
                    set(h_smart(6,2:2:12),'Color','r');
                    set(h_smart(1,1:2:12),'LineStyle','-','Color','b');      % Top whisker line
                    set(h_smart(1,2:2:12),'LineStyle','-','Color','r');      % Top whisker line
                    set(h_smart(2,1:2:12),'LineStyle','-','Color','b');      % Bottom whisker line
                    set(h_smart(2,2:2:12),'LineStyle','-','Color','r');      % Bottom whisker line
                    set(h_smart(3,1:2:12),'Color','b');          % Top whisker bar
                    set(h_smart(3,2:2:12),'Color','r');
                    set(h_smart(4,1:2:12),'Color','b');          % Bottom whisker bar
                    set(h_smart(4,2:2:12),'Color','r');
                    
                    if size(h_smart,1) > 6
                        set(h_smart(7,1:2:12),'MarkerEdgeColor','b','MarkerSize',4); 
                        set(h_smart(7,2:2:12),'MarkerEdgeColor','r','MarkerSize',4); 
                    end
                    
                    h_axes = N_plot(11);                            
                    %axis(h_axes,[0.5 6.5 30  90]);
                    xlim(N_plot(11),[0.5 6.5]);
                    ylim([30 90]);
                    set(h_axes,'YGrid','on')
                    set(h_axes,'YTick',30:20:90);
                    set(h_axes,'YtickLabel',30:20:90);
                    Xtick_pos = [];
                    for i = 1:2:length(group_positions)
                        Xtick_pos = [Xtick_pos (group_positions(i) + group_positions(i+1))/2];
                    end
                    set(h_axes,'Xtick',Xtick_pos);
                    ylabel('Accuracy (%)','FontSize',paper_font_size-1);
                    xlabel({'Subjects (Calibration modes)'});
                    title('Offline Calibration Performance','FontSize',paper_font_size-1);
                    sigstar(sig_intervals,p_values);
                    set(gca,'Position',ax11pos_orig)
                    set(gca,'Box','off')
   
    
    [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([h_conv_roc, h_smart_roc],{'Fixed window','Adaptive window'},'nrow',2, 'ref',N_plot(11),...
                                            'anchor',[5 5],'buffer',[0 -5],'box','off','xscale',0.5,'padding',[0 10 0],'fontsize',paper_font_size-1);
end
%% Plot accuracy for all subjects- NEW format
if plot_offline_performance_new == 1
     figure('Position',[1050 1100 3.5*116 2.5*116]); 
    T_plot = tight_subplot(1,1,[0.05 0.02],[0.15 0.1],[0.1 0.01]);
    Subject_names = {'LSGR','PLSH','PLSH','ERWS','BNBO','BNBO'};
    Cond_nos = [3 3 1 3 3 1];
    %group_conv_perf = [];
    %group_conv_labels = [];
    %group_smart_perf = [];
    %group_smart_labels = [];
    group_conv_smart_perf = [];
    group_conv_smart_labels = [];
    sig_intervals = cell(1);
    p_values = [];
    
    for subj_n = 1:length(Subject_names)
                    Sess_num = '2';
                    Block_num = 160;
                    cond_n = Cond_nos(subj_n);
                    if subj_n == 1
                        Sess_num = '2b';
                        Block_num = 140;
                    end
                        
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(Sess_num) '\'];
                    Performance = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
                    %chance_level_smart = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_chance_level_smart2.mat']);
                    Performance_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);
                    %chance_level_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_chance_level_conventional2.mat']);
                    
                    offline_conv = (Performance_conv.All_eeg_accur{Performance_conv.conv_opt_wl_ind}(1,:))';
                    %chance_conv = chance_level_conv.Chance_accuracy_permutations_labels;
                    %group_conv_perf = [group_conv_perf; [offline_conv; chance_conv]];
                    %group_conv_labels = [group_conv_labels; [subj_n*ones(size(offline_conv,1),1); (subj_n+0.5)*ones(size(chance_conv,1),1)]];
                    
                    offline_smart = (Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:))';
                    %chance_smart = chance_level_smart.Chance_accuracy_permutations_labels;
                    %group_smart_perf = [group_smart_perf; [offline_smart; chance_smart]];
                    %group_smart_labels = [group_smart_labels; [subj_n*ones(size(offline_smart,1),1); (subj_n+0.5)*ones(size(chance_smart,1),1)]];
                    group_conv_smart_perf = [group_conv_smart_perf; [offline_conv;offline_smart]];
                    group_conv_smart_labels = [group_conv_smart_labels; [subj_n*ones(size(offline_conv,1),1); (subj_n+0.5)*ones(size(offline_smart,1),1)]];
                    
                     % Significance tests
                    [pwilcoxon,h,stats] = ranksum(offline_smart,offline_conv,'alpha',0.05,'tail','right');
                    if h == 1
                        sig_intervals(end+1) = {[subj_n-0.15 subj_n+0.15]};
                        if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                            p_values = [p_values 0.05];
                        elseif (pwilcoxon <= 0.01)
                            p_values = [p_values 0.01];
                        end
                     end
                    
    end
                    sig_intervals(1) = [];
                    group_names = {{'  S1';''; '  S2';''; '  S2';'';'  S3';'';'  S4';'';'  S4';''},{' (AT)';'';' (BD)';'';' (AT)';'';'  (AT)';'';'  (BD)';'';'  (AT)';''}};
                    group_positions = [0.85 1.15 1.85 2.15 2.85 3.15 3.85 4.15 4.85 5.15 5.85 6.15];
                    % Adaptive window
                    axes(T_plot); hold on;                    
                    h_smart = boxplot(group_conv_smart_perf,group_conv_smart_labels,'labels', group_names, 'positions',group_positions,...
                                                        'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
                    set(h_smart,'LineWidth',1);
                    set(h_smart(5,1:2:12),'Color','b');         % Box
                    set(h_smart(5,2:2:12),'Color','r');
                    set(h_smart(6,1:2:12),'Color','b');          % Median
                    set(h_smart(6,2:2:12),'Color','r');
                    set(h_smart(1,1:2:12),'LineStyle','-','Color','b');      % Top whisker line
                    set(h_smart(1,2:2:12),'LineStyle','-','Color','r');      % Top whisker line
                    set(h_smart(2,1:2:12),'LineStyle','-','Color','b');      % Bottom whisker line
                    set(h_smart(2,2:2:12),'LineStyle','-','Color','r');      % Bottom whisker line
                    set(h_smart(3,1:2:12),'Color','b');          % Top whisker bar
                    set(h_smart(3,2:2:12),'Color','r');
                    set(h_smart(4,1:2:12),'Color','b');          % Bottom whisker bar
                    set(h_smart(4,2:2:12),'Color','r');
                    
                    if size(h_smart,1) > 6
                        set(h_smart(7,1:2:12),'MarkerEdgeColor','b','MarkerSize',4); 
                        set(h_smart(7,2:2:12),'MarkerEdgeColor','r','MarkerSize',4); 
                    end
                    
                    h_axes = gca;                            
                    axis(h_axes,[0.5 6.5 30  90]);
                    set(h_axes,'YGrid','on')
                    set(h_axes,'YTick',30:20:90);
                    Xtick_pos = [];
                    for i = 1:2:length(group_positions)
                        Xtick_pos = [Xtick_pos (group_positions(i) + group_positions(i+1))/2];
                    end
                    set(h_axes,'YtickLabel',30:20:90);
                    set(h_axes,'Xtick',Xtick_pos);
                    
%                     h_colors = findobj(gca,'Tag','Box');
%                     for j = length(h_colors):-1:1
%                         if mod(j,2)   % odd number
%                             h_patch_w = patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),[0.6 0.6 0.6]);
%                         else
%                             h_patch_smart = patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),[1 1 1]);
%                         end
%                     end     
                    ylabel('Accuracy (%)','FontSize',10);
                    xlabel({' ';'Subjects (Calibration modes)'});
                    title('Offline Calibration Performance','FontSize',10);
                    %set(T_plot(subj_n),'YGrid','on','GridLineStyle',':');
                    %text(0.9,25,sprintf('(%.2fs)',Performance_conv.conv_window_length),'FontSize',10);
                    %text(2.45,25,sprintf('(%.2fs)',Performance.smart_window_length),'FontSize',10);                 
                    sigstar(sig_intervals,p_values);
end

%     fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
%     %hgexport(fig,'-clipboard',style,'applystyle', true);
%     drawnow;

%     response = input('Save figure to folder [y/n]: ','s');
%     if strcmp(response,'y')
%         tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.tif'];
%         fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.fig'];
%         print('-dtiff', '-r300', tiff_filename); 
%         saveas(gcf,fig_filename);
%         im = imread(tiff_filename, 'tiff' );
%         [im_hatch,colorlist] = applyhatch_pluscolor(im,'\w',0,0,[0 0 1; 1 0 0],300,3,2);
%         imwrite(im_hatch,tiff_filename,'tiff');        
%     else
%         disp('Save figure aborted');
%     end

