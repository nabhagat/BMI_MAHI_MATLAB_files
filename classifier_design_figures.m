% Plot figures for classifier design steps
clear;

% Select figures to plot
plot_erp_image = 0; 
plot_electrode_locations = 0;
plot_scalp_map = 0;
plot_eeg_emg_latencies = 0;
plot_feature_space = 0;
plot_offline_performance = 0;

% Subject Details
Subject_name = 'BNBO';
Sess_num = '2';
Cond_num = 1;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
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
    
%% Plot ERP image plot for classifier channels after sorting
if plot_erp_image == 1
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
figure('Position',[50 300 8*116 3*116]); 
T_plot = tight_subplot(1,length(classchannels)+1,[0.08 0.05],[0.15 0.1],[0.1 0.1]);
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
                                                                                                            'limits',[-2.5 1.5],'caxis',caxis_range,...%'vert',[-2 0.5],'horz',[132],...
                                                                                                            'cbar','off','cbar_title','\muV',...
                                                                                                            'noxlabel','on','avg_type',avergae_type,...
                                                                                                            'img_trialax_label',[],'img_trialax_ticks',[20 40 60 80 100 120 140 160]);
                                                                                                        
    img_handle = axhndls(1);
    img_xlab_handle = get(img_handle,'xlabel');
    img_ylab_handle = get(img_handle,'ylabel');
    img_title_handle = get(img_handle,'title');
    set(img_title_handle,'FontSize',10,'FontWeight','normal');
    set(img_handle,'YLim',[1 size(move_epochs_s,1)])
    set(img_handle,'TickLength',[0.03 0.025]);
    
    if ch == 1
        set(img_xlab_handle,'String','Time(sec.)','FontSize',10);
        set(img_ylab_handle,'String','Sorted Single Trials','FontSize',10);   
    end
    img_handle_children = get(img_handle,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
    set(img_handle_children(1),'LineWidth',1.5);
end
     axes(T_plot(length(classchannels) + 1));
    [outdata,outvar,outtrials,limits,axhndls, ...
                    erp,amps,cohers,cohsig,ampsig,outamps,...
                    phsangls,phsamp,sortidx,erpsig] = erpimage(sorted_move_ch_avg_ini',[],move_erp_time,'Spatial Average',average_width,decimate,...
                                                                                                        'limits',[-2.5 1.5],'caxis',caxis_range,'vert',[-1.5],'horz',length(good_move_trials)+1,...
                                                                                                        'cbar','on','cbar_title','\muV',...
                                                                                                        'noxlabel','on','avg_type',avergae_type,...
                                                                                                        'img_trialax_label',[],'img_trialax_ticks',[20 40 60 80 100 120 140 160]);

    img_handle = axhndls(1);
    img_title_handle = get(img_handle,'title');
    set(img_title_handle,'FontSize',10,'FontWeight','normal');
    set(img_handle,'TickLength',[0.03 0.025]);    
    img_handle_children = get(img_handle,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
    set(img_handle_children(1),'LineWidth',1.5);
    set(img_handle_children(2),'LineWidth',2,'Color',[0 0 0]);
    set(img_handle_children(3),'LineWidth',1.5,'Color',[0 0 0]);
    
    cbar_handle = axhndls(2);
    cbar_child_handle = get(cbar_handle,'child');
    bar_pos = get(cbar_handle,'Position');
    set(cbar_handle,'Position',[bar_pos(1) bar_pos(2) bar_pos(3)/2 bar_pos(4)/3]);
    ytickpos = get(cbar_handle,'Ytick');
    set(cbar_handle,'Ytick',[ytickpos(1) ytickpos(3) ytickpos(end)]);
    set(cbar_handle,'YtickLabel',[-10 0 10]);
%     if ch ~= length(classchannels)
%          set(cbar_handle,'Visible','off');
%          set(cbar_child_handle,'Visible','off');         
%     end

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
    print('-dtiff', '-r300', tiff_filename); 
    saveas(gcf,fig_filename);
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
    figure('Position',[50 300 6 *116 5*116]); 
    T_plot = tight_subplot(2,3,[0.15 0.1],[0.15 0.1],[0.1 0.01]);
    
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
    set(hscatter_conv(1),'DisplayName','Go');
    set(hscatter_conv(2),'DisplayName','No-go');
    set(hscatter_conv(1),'LineWidth',1);
    set(hscatter_conv(2),'LineWidth',1);
    title('Fixed window','FontSize',10);
    leg_scatter= findobj(gcf,'tag','legend'); %set(leg,'location','northwest');
    
    

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
    
    leg_auc = legend([hopt_auc h_smart_roc hconv_auc hsmart_auc],{['optimal window' char(10)  'length (wl_o)'],'ROC for wl_o', 'Fixed window', 'Adaptive window'});
    
    leg_auc_pos = get(leg_auc,'Position');
    axes6pos = get(T_plot(6),'Position');
    set(leg_auc,'position',[axes6pos(1) - (leg_auc_pos(3) - axes6pos(3)) axes6pos(2)+axes6pos(4)-leg_auc_pos(4) leg_auc_pos(3) leg_auc_pos(4)]);
    mod_leg_auc_pos = get(leg_auc,'Position');
    leg_scatter_pos = get(leg_scatter,'Position');
    set(leg_scatter,'position',[mod_leg_auc_pos(1) mod_leg_auc_pos(2)-leg_scatter_pos(4) leg_scatter_pos(3) leg_scatter_pos(4)]);
    set(leg_auc,'Box','off');
    set(leg_scatter,'Box','off');
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
     figure('Position',[50 300 6*116 4*116]); 
    T_plot = tight_subplot(2,3,[0.05 0.05],[0.15 0.1],[0.1 0.01]);
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
                    Performance_conv = importdata([folder_path  Subject_names{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(cond_n) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);

                    axes(T_plot(subj_n));
                    group_names = {{'Fixed';'Adaptive'},{'Window';'Window'}};
                    offline_conv = Performance_conv.All_eeg_accur{Performance_conv.conv_opt_wl_ind}(1,:);
                    offline_smart = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
                    h_offline_acc = boxplot([offline_conv' offline_smart'],'labels',group_names,'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','+'); % symbol - Outliers take same color as box
                    set(h_offline_acc,'LineWidth',2);
                    set(h_offline_acc(5,1),'Color','b'); set(h_offline_acc(6,1),'Color','b');
                    set(h_offline_acc(5,2),'Color','r'); set(h_offline_acc(6,2),'Color','r');
                    if size(h_offline_acc,1) > 6
                        set(h_offline_acc(7,1),'MarkerEdgeColor','b'); set(h_offline_acc(7,2),'MarkerEdgeColor','r');       % Color of outliers
                    end
                    
                    line([0 3],[50 50],'LineStyle','--', 'LineWidth',1.5,'Color','k');
                    ylim([20 90]);
                    %xlim([1 5]);
                    set(gca,'YTick',[20:10:100]);
                    set(gca,'YTickLabel','');
                    if (subj_n == 1) || (subj_n == 4)
                        set(gca,'YTick',[20:10:100]);
                        set(gca,'YTickLabel',{'' '30' '' '50' '' '70' '' '90' '100'});
                        %ylabel('Classification Accuracy (%)','FontSize',myfont_size,'FontWeight',myfont_weight);
                        ylabel('Offline Accuracy (%)','FontSize',10);
                    end
                    
                    if (subj_n == 1) || (subj_n == 2) || (subj_n == 3)
                        set(gca,'XTickLabel',{' '})                                                                                                         % Hide group label names
                    end
                    
                    switch subj_n
                        case 1
                            title_str = 'S1, TR';
                        case 2
                            title_str = 'S2, TR';
                        case 3
                            title_str = 'S2, BD';
                        case 4
                            title_str = 'S3, TR';
                        case 5
                            title_str = 'S4, TR';
                        case 6
                            title_str = 'S4, BD';
                    end
                    title(title_str,'FontSize',10);
                    set(T_plot(subj_n),'YGrid','on','GridLineStyle',':');
                    text(0.75,25,sprintf('(%.2fs)',Performance_conv.conv_window_length),'FontSize',10);
                    text(1.75,25,sprintf('(%.2fs)',Performance.smart_window_length),'FontSize',10);
                    
              
    end
%% Expand axes to fill figure
    fig = gcf;
    style = hgexport('factorystyle');
    style.Bounds = 'tight';
    hgexport(fig,'-clipboard',style,'applystyle', true);
    drawnow;

    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
         tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.tif'];
         fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_offline_accuracy.fig'];
        print('-dtiff', '-r300', tiff_filename); 
        saveas(gcf,fig_filename);
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



