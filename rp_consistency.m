%% Program to check consistency of MRCPs by computing cross-correlation
% Date modified: 2-17-2015
clear;
paper_font_size = 10;
% Subject Details
Subject_name = 'LSGR'; subj_n = 1; % change1
Sess_num = '2b';
Block_num = 140;

all_Cond_num = [3 3];  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
cloop_ses = [4 5];
mu_chance = 0; %0.33;
correlation_lags = 40;
plot_correlation_analysis = 1;

if plot_correlation_analysis == 1
        figure('Position',[700 1100 3.5*116 3.5*116]);     % [left bottom width height]
        U_plot = tight_subplot(3,2,[0.05 0.02],[0.12 0.1],[0.15 0.1]);
        height_inc = 0.05; 
        height_shift = 0.005;
        
        axes(U_plot(1));
        pos_p = get(gca,'Position'); 
        set(gca,'Position',[pos_p(1) pos_p(2)-height_inc pos_p(3) pos_p(4)]);
        
        axes(U_plot(2));
        pos_p = get(gca,'Position'); 
        set(gca,'Position',[pos_p(1) pos_p(2)-height_inc pos_p(3) pos_p(4)]);
        %set(gca,'Position',[pos_p(1) pos_p(2)-height_inc+0.01 pos_p(3) pos_p(4)+height_inc-0.01]);
        
        axes(U_plot(5));
        pos_p1 = get(gca,'Position'); 
        set(gca,'Position',[pos_p1(1) pos_p1(2) pos_p1(3) pos_p1(4)-2*height_inc]);
        pos_p1 = get(gca,'Position');
        
        axes(U_plot(3));
        pos_p2 = get(gca,'Position'); 
        set(gca,'Position',[pos_p2(1), pos_p1(2)+pos_p1(4)+height_shift, pos_p2(3), pos_p2(4) + (pos_p2(2) - (pos_p1(2)+pos_p1(4)+3*height_inc))]);  
        %set(gca,'Position',[pos_p2(1), pos_p1(2)+pos_p1(4)+height_shift, pos_p2(3), pos_p2(4)]);  
        
        axes(U_plot(6));
        pos_p1 = get(gca,'Position'); 
        set(gca,'Position',[pos_p1(1) pos_p1(2) pos_p1(3) pos_p1(4)-2*height_inc]);
        pos_p1 = get(gca,'Position');
        
        axes(U_plot(4));
        pos_p2 = get(gca,'Position'); 
        set(gca,'Position',[pos_p2(1), pos_p1(2)+pos_p1(4)+height_shift, pos_p2(3), pos_p2(4) + (pos_p2(2) - (pos_p1(2)+pos_p1(4)+3*height_inc))]);    
        
for n = 1:length(cloop_ses)
    move_epochs_s = [];
    Xcorr_trials = [];
    closeloop_Sess_num = cloop_ses(n);  
    Cond_num = all_Cond_num(n);
    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; 
    closeloop_folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; 

    load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_average_causal.mat']);
    load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);
    load([closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_cloop_eeg_epochs.mat']);                
    cl_ses_data = dlmread([closeloop_folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_cloop_statistics.csv'],',',7,1); 
    
    raw_Fs = Average.Fs_eeg;
    resamp_Fs = 20;
    %downsamp_factor = raw_Fs/resamp_Fs;

     [no_epochs,no_datapts,no_channels] = size(Average.move_epochs); %#ok<*ASGLU> 
     for k = 1:no_channels
            move_epochs_s(:,:,k) = downsample(Average.move_epochs(:,:,k)',Average.Fs_eeg/resamp_Fs)';
    end
    %move_avg_channels_s = downsample(Average.move_avg_channels',Average.Fs_eeg/resamp_Fs)'; % Average of epochs for each channel
    move_erp_time_s = downsample(Average.move_erp_time,Average.Fs_eeg/resamp_Fs);
    %move_average_classchannels = move_avg_channels_s(Performance.classchannels,find(move_erp_time_s == -2.0):find(move_erp_time_s == 1.0)); % Average of epochs for classifier channels
    classchannels = Performance.classchannels;
    
    % Compute Spatial Average (Mean Filtering) of chosen classifier channels
    move_ch_avg_ini = mean(move_epochs_s(Performance.good_move_trials,:,classchannels),3);
    grand_move_spatial_avg = mean(move_ch_avg_ini(:,find(move_erp_time_s == -2):find(move_erp_time_s == 1)),1);   % Computed for calibration data

    % Get EEG and spatial average for trials when trial was valid and successful i..e. intent was detected 
    ind_success_valid_trials = find((cl_ses_data(:,4) == 1) & (cl_ses_data(:,5) == 1)); % col 4 - valid or catch; col 5 - Intent detected
    success_Intent_EEG_epochs_session = Intent_EEG_epochs_session(:,:,ind_success_valid_trials);
    % Further remove trials for which EEG epochs were unable to extract
    corrupt_trials = [];
    for time_no = 1:size(success_Intent_EEG_epochs_session,3)
        if isempty(find(success_Intent_EEG_epochs_session(1,:,time_no) == -2,1))
            corrupt_trials = [corrupt_trials, time_no];
        end
    end
    if ~isempty(corrupt_trials)
        corrupt_response = input([num2str(length(corrupt_trials)) ' corrupted trail(s) will be deleted, Do you wish to proceed? [y/n]: '],'s');
        if strcmp(corrupt_response,'y')
            success_Intent_EEG_epochs_session(:,:,corrupt_trials) = [];
        end
    end
    
    cloop_ses_spatial_avgs = squeeze(success_Intent_EEG_epochs_session(2,find(success_Intent_EEG_epochs_session(1,:,2) == -2):find(success_Intent_EEG_epochs_session(1,:,2) == 1),:))';
    epoch_time = success_Intent_EEG_epochs_session(1,find(success_Intent_EEG_epochs_session(1,:,1) == -2):find(success_Intent_EEG_epochs_session(1,:,1) == 1),1);
    
    % Calculate Covariance between grand_spatial_average (grand_move_spatial_avg) and spatial_average_for_single_trials (cloop_ses_spatial_avgs) 
    for trial_no = 1:size(cloop_ses_spatial_avgs,1)
        [Xcorr_trials(trial_no,:),lags] = xcorr(grand_move_spatial_avg,cloop_ses_spatial_avgs(trial_no,:),correlation_lags,'coeff');  % 30 lags = 1.5 secs
    end
    
    % t-tests
    [H,P,~,Stat] = ttest(Xcorr_trials,mu_chance,'Tail','right','Alpha',0.05);
    if isempty(find(H==1,1))
        mu_chance = 0;
        [H,P,~,Stat] = ttest(Xcorr_trials,mu_chance,'Tail','right','Alpha',0.05);
    end
    
    if isempty(find(H==1,1))
        no_signifance = 1;
        region_of_significance = [2 2];
    else 
        no_signifance = 0;
        region_of_significance = [lags(find(H==1,1,'first')), lags(find(H==1,1,'last'))]/resamp_Fs;
    end
    
    normalized_grand_move_spatial_avg = zscore(grand_move_spatial_avg);
    normalized_cloop_ses_spatial_avgs = zscore(cloop_ses_spatial_avgs')';
  
    axes(U_plot(n)); hold on;
    h_spatial = plot(epoch_time',normalized_cloop_ses_spatial_avgs','LineWidth',1,'Color',[0.7 0.7 0.7]);
    h_gavg = plot(epoch_time,normalized_grand_move_spatial_avg,'LineWidth',1,'Color','k');
    axis([-2 1 -4 2]);
    set(gca,'Xtick',[-2 -1 0 1],'XtickLabel',{' ' ' ' 'Intent or MO' ' '},'xgrid','on','FontSize',paper_font_size -1);
    %text(-0.3,4.5,'detected');
    set(gca,'Ydir','reverse');
    set(gca,'Ytick',[-2 0 2],'YtickLabel',{' '},'ygrid','on','box','on');
    line([0 0],[-5.5 3],'Color','k','LineWidth',0.5,'LineStyle','--');
    %ylim([-5.5 3]);
    
    %hxlab = xlabel('Time (sec.)'); 
    %pos_hxlab = get(hxlab,'Position');
    %set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) + 0.6) pos_hxlab(3)]);
    
    
    if Cond_num == 1
            title({['Day ' num2str(closeloop_Sess_num)]},'FontSize',paper_font_size-1);
        elseif Cond_num == 3
            %title({['Day ' num2str(closeloop_Sess_num) ', S' num2str(subj_n) ' (AT)'];'Spatial Averages'});
            title({['Day ' num2str(closeloop_Sess_num)]},'FontSize',paper_font_size-1);
    end
        
    if closeloop_Sess_num == 4
        ylabel({'Standardized';'Amplitude'},'FontSize',paper_font_size-1);
        %plot_leg = legend([h_gavg h_spatial(1)],'Mean (Offline training trials)', 'Single-trials (Online BMI Control)','location','NorthWest','Orientation','vertical');
        %set(plot_leg,'FontSize',8,'box','off');
        %pos_leg = get(plot_leg,'Position');
        %axpos = get(gca,'Position');
        %set(plot_leg,'Position',[axpos(1)-0.005 (axpos(2)+axpos(4) - pos_leg(4)) pos_leg(3) pos_leg(4)]);
        axes_pos = get(gca,'Position');
        if Cond_num == 1
            text(-1.95,-3.5, ['S' num2str(subj_n) ' (\bfBD\rm)'],'FontSize',paper_font_size-1);
        else
            text(-1.95,-3.5, ['S' num2str(subj_n) ' (AT)'],'FontSize',paper_font_size-1);
        end
        [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([h_gavg, h_spatial(1)],{'Grand Saptial Avg (Offline)','Single-trial Spatial Avg (Closed-loop)'},'nrow',2, 'ref',U_plot(1),...
                                            'anchor',[1 7],'buffer',[0 20],'box','off','xscale',0.3,'padding',[0 5 5]);
                                        
    elseif closeloop_Sess_num == 5
        axes_pos = get(gca,'Position'); %[lower bottom width height]
        axes_ylim = get(gca,'Ylim');
        annotate_length = (2*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
        annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.025) (axes_pos(1)+axes_pos(3)+0.025)],...
        [axes_pos(2) (axes_pos(2) + annotate_length)],'LineWidth',2);
    
        axes_xlim = get(gca,'Xlim');
        annotate_length = (axes_pos(3))/(axes_xlim(2) - axes_xlim(1));
        annotation(gcf,'line', [axes_pos(1) (axes_pos(1) + annotate_length)],...
        [(axes_pos(2)-0.025) (axes_pos(2)-0.025)],'LineWidth',2);
        annotation('textbox',[(axes_pos(1)+0.02) (axes_pos(2)-0.12) annotate_length 0.1],'String','1s','EdgeColor','none','FontSize',paper_font_size-1);
         if Cond_num == 1
            text(-1.95,-3.5, ['S' num2str(subj_n) ' (\bfBD\rm)'],'FontSize',paper_font_size-1);
        else
            text(-1.95,-3.5, ['S' num2str(subj_n) ' (AT)'],'FontSize',paper_font_size-1);
        end
    end
    
        
        
        
        
    %else
%         set(gca,'Ytick',[-2 0 2],'YtickLabel',{'-2' '0' '2'},'ygrid','on','box','on');
%         if Cond_num == 1
%             title({['Day ' num2str(closeloop_Sess_num) ', S' num2str(subj_n) ' (BD)'];'Spatial Averages'});
%         elseif Cond_num == 3
%             title({['Day ' num2str(closeloop_Sess_num) ', S' num2str(subj_n) ' (AT)'];'Spatial Averages'});
%         end
%     end
    
    axes(U_plot(n+2));
    average_width = 2;  % Vertical moveing average filter; takes 2 trials above and below the trial of interest
    avergae_type = 'boxcar'; % Avoid gaussian because avewidth of 1, applies a vertical window of 7 trials
    decimate = [];
    caxis_range = [-1 1];

    erpdata = Xcorr_trials'; % erpdata = frames x trials

    if closeloop_Sess_num == 4
        [outdata,outvar,outtrials,limits,axhndls, ...
                            erp,amps,cohers,cohsig,ampsig,outamps,...
                            phsangls,phsamp,sortidx,erpsig] = erpimage(erpdata,[],lags./resamp_Fs,'Cross-correlation \itR_{XY}\rm(\epsilon)',average_width,decimate,...
                                                                                                            'limits',[lags(1)/resamp_Fs lags(end)/resamp_Fs],'caxis',caxis_range,'vert',region_of_significance,...%'horz',length(good_move_trials)+1,...
                                                                                                            'cbar','off','cbar_title','\rho (a.u.)',...
                                                                                                            'noxlabel','on','avg_type',avergae_type,...
                                                                                                            'img_trialax_label',[],'img_trialax_ticks',[50 100]);

        size_erpimage4 = size(Xcorr_trials,1);
        img_handle4 = axhndls(1);
        img_xlab_handle = get(img_handle4,'xlabel');
        img_ylab_handle = get(img_handle4,'ylabel');
        img_title_handle = get(img_handle4,'title');
        img_title_pos = get(img_title_handle,'Position');
        set(img_title_handle,'FontSize',paper_font_size-1,'FontWeight','normal','Position',[img_title_pos(1)+2.25 img_title_pos(2)-5 img_title_pos(3)]);
        set(img_handle4,'YLim',[1 size(Xcorr_trials,1)])
        set(img_handle4,'TickLength',[0.03 0.025]);
        set(img_handle4,'Xtick',[-1 0 1]);
        set(img_handle4,'XTickLabel',{''},'XGrid','on');
        %set(img_xlab_handle,'String','Lags (sec.)','FontSize',10);
        set(img_ylab_handle,'String',{'Single-trials';'Spatial Avg'},'FontSize',paper_font_size-1);   
        ylab_pos = get(img_ylab_handle,'Position');
        set(img_ylab_handle,'Position',[ylab_pos(1)+0.3 ylab_pos(2)+0.01 ylab_pos(3)])
        set(img_handle4,'Ytick',[50 100], 'YTickLabel',{'50' '100'});
        img_handle_children = get(img_handle4,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
        set(img_handle_children(1),'LineWidth',0.5,'LineStyle','--','Visible','off');
        set(img_handle_children(2),'LineWidth',1,'Color',[0 0 1],'LineStyle','-');
        set(img_handle_children(3),'LineWidth',1,'Color',[0 0 1],'LineStyle','-');    
        
    else
        [outdata,outvar,outtrials,limits,axhndls, ...
                            erp,amps,cohers,cohsig,ampsig,outamps,...
                            phsangls,phsamp,sortidx,erpsig] = erpimage(erpdata,[],lags./resamp_Fs,' ',average_width,decimate,...
                                                                                                            'limits',[lags(1)/resamp_Fs lags(end)/resamp_Fs],'caxis',caxis_range,'vert',region_of_significance,...%'horz',length(good_move_trials)+1,...
                                                                                                            'cbar','on','cbar_title','a.u.',...
                                                                                                            'noxlabel','on','avg_type',avergae_type,...
                                                                                                            'img_trialax_label',[],'img_trialax_ticks',[]);

        size_erpimage5 = size(Xcorr_trials,1);
        img_handle5 = axhndls(1);
        %img_xlab_handle = get(img_handle5,'xlabel');
        %img_ylab_handle = get(img_handle5,'ylabel');
        %img_title_handle = get(img_handle5,'title');
        %set(img_title_handle,'FontSize',paper_font_size-1,'FontWeight','normal');
        set(img_handle5,'YLim',[1 size(Xcorr_trials,1)])
        %set(img_handle5,'YTickLabel','');
        set(img_handle5,'TickLength',[0.03 0.025]);
        set(img_handle5,'Xtick',[-1 0 1]);
        set(img_handle5,'XTickLabel',{''},'XGrid','on','YtickLabel',' ');
        
        img_handle_children = get(img_handle5,'Children'); % [Movement_onset_line horizontanl_line vertical_line colormap]
        set(img_handle_children(1),'LineWidth',0.5,'LineStyle','--','Visible','off');
        set(img_handle_children(2),'LineWidth',1,'Color',[0 0 1],'LineStyle','-');
        set(img_handle_children(3),'LineWidth',1,'Color',[0 0 1],'LineStyle','-');
        
        if size_erpimage5 > size_erpimage4
             set(img_handle4,'YLim',[1 size_erpimage5],'Ytick',[50 100],'YtickLabel',{'50' '100'})
             curr_title_pos = get(img_title_handle,'Position');
             set(img_title_handle,'Position',[curr_title_pos(1) size_erpimage5 curr_title_pos(3)]);
             
            old_ylab_pos = get(img_ylab_handle,'Position');
            set(img_ylab_handle,'Position',[old_ylab_pos(1) old_ylab_pos(2)+10 old_ylab_pos(3)]);
        elseif size_erpimage4 > size_erpimage5
            set(img_handle5,'YLim',[1 size_erpimage4],'Ytick',[50 100],'YtickLabel',{' '})
        end
            
        
        
        cbar_handle = axhndls(2);
        cbar_child_handle = get(cbar_handle,'child');
        bar_pos = get(cbar_handle,'Position');
        set(cbar_handle,'Position',[bar_pos(1)+0.02 bar_pos(2) bar_pos(3)/2 bar_pos(4)/2]);
        %set(get(cbar_handle,'Title'),'String',{'\it \fontsize{8} R_{XY} \rm'; '(a.u.)'})
        ytickpos = get(cbar_handle,'Ytick');
        set(cbar_handle,'Ytick',[ytickpos(1) ytickpos(3) ytickpos(end)]);
        set(cbar_handle,'YtickLabel',[-1 0 1]);
        %set(cbar_handle,'YDir','reverse');
    
    end
        
    
    axes(U_plot(n+4));
    plot(lags./resamp_Fs,Stat.tstat,'LineWidth',1,'Color','k');
    set(gca,'Xtick',[-1 0 1],'XtickLabel',{ '-1' '0' '1' },'xgrid','on','FontSize',paper_font_size-1);
    if closeloop_Sess_num == 4
        max_range4 = ceil(max(Stat.tstat));
        max_range = max_range4;
         ylabel({'t-score'},'FontSize',paper_font_size-1);
    else
        max_range5 = ceil(max(Stat.tstat));
        if max_range5 > max_range4
            max_range = max_range5;
            ylim(U_plot(5), [0, max_range + 2]); xlim([lags(1)/resamp_Fs lags(end)/resamp_Fs]);
        elseif max_range5 < max_range4
            max_range = max_range4;
        end
    end      
    ylim([0, max_range + 2]); xlim([lags(1)/resamp_Fs lags(end)/resamp_Fs]);
    line([region_of_significance(1) region_of_significance(1)],[0 max_range],'LineWidth',1,'Color','b');
    line([region_of_significance(2) region_of_significance(2)],[0 max_range],'LineWidth',1,'Color','b');   
    
    xlabel('Lags (\epsilon) (sec.)','FontSize',paper_font_size-1); 
    set(U_plot(5),'Ytick',[0 max_range],'YtickLabel',{'0' num2str(max_range)},'ygrid','on','box','on');
    set(U_plot(6),'Ytick',[0 max_range],'YtickLabel',{' '},'ygrid','on','box','on');
    h_stars = sigstar({region_of_significance},0.05);   
    set(h_stars(1),'Color','b','Linewidth',1);
    set(h_stars(2),'Color','b');
end    

%annotation('textbox',[0 0 0.2 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
% Expand axes to fill figure
% fig = gcf;
% style = hgexport('factorystyle');
% style.Bounds = 'tight';
% hgexport(fig,'-clipboard',style,'applystyle', true);
% drawnow;


%  response = input('Save figure to folder [y/n]: ','s');
%     if strcmp(response,'y')
%         tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_mrcp_consistency.tif'];
%         fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_mrcp_consistency.fig'];
% %         print('-dtiff', '-r600', tiff_filename); 
% %         saveas(gcf,fig_filename);
%     else
%         disp('Save figure aborted');
%     end
end
%%  Histograms for feature vectors

% subplot(2,2,1); hold on
% hist(session_feature_vec(1,:),10);
% h1 = findobj(gca,'Type','patch');
% title('Slope','FontSize',14);
% 
% subplot(2,2,2); hold on
% hist(session_feature_vec(2,:),10);
% h2 = findobj(gca,'Type','patch');
% title('Negative Peak','FontSize',14);
% 
% subplot(2,2,3); hold on
% hist(session_feature_vec(3,:),10);
% h3 = findobj(gca,'Type','patch');
% title('Area Under the Curve','FontSize',14);
% 
% subplot(2,2,4); hold on
% hist(session_feature_vec(4,:),10);
% h4 = findobj(gca,'Type','patch');
% title('Mahalanobis Distance','FontSize',14);
% 
% hist_color = [0 0 1];
% set(h1(3),'FaceColor',hist_color,'EdgeColor','b','FaceAlpha',0,'LineWidth',2)
% set(h2(3),'FaceColor',hist_color,'EdgeColor','b','FaceAlpha',0,'LineWidth',2)
% set(h3(3),'FaceColor',hist_color,'EdgeColor','b','FaceAlpha',0,'LineWidth',2)
% set(h4(3),'FaceColor',hist_color,'EdgeColor','b','FaceAlpha',0,'LineWidth',2)
% 
% hist_color = [0 1 0];
% set(h1(2),'FaceColor',hist_color,'EdgeColor','g','FaceAlpha',0,'LineWidth',2)
% set(h2(2),'FaceColor',hist_color,'EdgeColor','g','FaceAlpha',0,'LineWidth',2)
% set(h3(2),'FaceColor',hist_color,'EdgeColor','g','FaceAlpha',0,'LineWidth',2)
% set(h4(2),'FaceColor',hist_color,'EdgeColor','g','FaceAlpha',0,'LineWidth',2)
% 
% hist_color = [1 0 0];
% set(h1(1),'FaceColor',hist_color,'EdgeColor','r','FaceAlpha',0,'LineWidth',2)
% set(h2(1),'FaceColor',hist_color,'EdgeColor','r','FaceAlpha',0,'LineWidth',2)
% set(h3(1),'FaceColor',hist_color,'EdgeColor','r','FaceAlpha',0,'LineWidth',2)
% set(h4(1),'FaceColor',hist_color,'EdgeColor','r','FaceAlpha',0,'LineWidth',2)





