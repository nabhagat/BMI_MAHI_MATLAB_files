% EEG Channels used for identifying MRCP
Channels_nos = [43, 9, 32, 10, 44, 13, 48, 14, 49, 15, 52, 19, ... 
                53, 20, 54]; % removed P-channels = [24, 57, 25, 58, 26]; removed F-channels = [4, 38, 5, 39, 6];    % 32 or 65 for FCz


%% Plot ERPs
if plot_ERPs == 1
    paper_font_size = 10;
    figure('Position',[1050 1300 3.5*116 2.5*116]); 
    %figure('units','normalized','outerposition',[0 0 1 1])
    T_plot = tight_subplot(3,5,[0.01 0.01],[0.15 0.01],[0.1 0.1]);
    hold on;
    plot_ind4 = 1;
    
    for ind4 = 1:length(Channels_nos)
        axes(T_plot(ind4));
%         if plot_ind4 == 5                 % Commented by Nikunj on Oct 31,2013
%             plot_ind4 = plot_ind4 + 1;
%         end
        %subplot(5,5,plot_ind4);
        hold on;
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:),'k','LineWidth',1.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:) - (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:),'r','LineWidth',2);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:)+ (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:) - (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         
%         jbfill(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),...
%            move_avg_channels(Channels_nos(ind4),:)- (move_SE_channels(Channels_nos(ind4),:)),[1 1 1],'k',0,0.3);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
%         jbfill(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:)+ (rest_SE_channels(Channels_nos(RPind),:)),...
%            rest_avg_channels(Channels_nos(RPind),:)- (rest_SE_channels(Channels_nos(RPind),:)),'r','k',0,0.3);
       
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),rest_time, rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,preprocessed_move_epochs(Channels_nos(RPind),:),'b',erp_time,standardize_move_epochs(Channels_nos(RPind),:),'k',erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        text(-2,-5,[EEG.chanlocs(Channels_nos(ind4)).labels],'Color','k','FontWeight','normal','FontSize',paper_font_size-1); % ', ' num2str(Channels_nos(ind4))
        set(gca,'YDir','reverse');
        %if max(abs(move_avg_channels(Channels_nos(RPind),:))) <= 6
%          if max((move_avg_channels(Channels_nos(ind4),:)+(move_SE_channels(Channels_nos(ind4),:)))) >= 6 || ...
%                  min((move_avg_channels(Channels_nos(ind4),:)-(move_SE_channels(Channels_nos(ind4),:)))) <= -6
%             axis([move_erp_time(1) move_erp_time(end) -15 15]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(gca,'FontWeight','bold');  
%             set(gca,'YTick',[-10 0 10]);
%             set(gca,'YTickLabel',{'-10'; '0'; '10'});
%         else
            axis([-2.5 1 -6 3]);
            %axis([move_erp_time(1) 1 -15 15]);
            set(gca,'YTick',[-5 0 2]);
            set(gca,'YTickLabel',{'-5'; '0'; '+2'},'FontWeight','normal','FontSize',paper_font_size-1);
%        end
        line([0 0],[-10 10],'Color','k','LineWidth',0.5,'LineStyle','--');  
        line([-2.5 1.5],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
        plot_ind4 = plot_ind4 + 1;
        
        if ind4 == 6
            set(gca,'XColor',[1 1 1],'YColor',[1 1 1])
            set(gca,'YtickLabel',' ');
             hylab = ylabel('MRCP Grand Average','FontSize',paper_font_size-1,'Color',[0 0 0]);
             pos_hylab = get(hylab,'Position');
             set(hylab,'Position',[pos_hylab(1) pos_hylab(2) pos_hylab(3)]);
        else
            set(gca,'Visible','off');
        end
        
    %    grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
     %   set(gca,'XTick',[-2 -1 0 1]);
     %   set(gca,'XTickLabel',{'-2';'-1'; '0';'1'});  
        
          
    end

    % subplot(4,3,5);
    % topoplot([],EEG.chanlocs,'style','blank','electrodes','labels','chaninfo',EEG.chaninfo);
    %subplot(5,5,8);
   
    
    axes(T_plot(11));
    set(gca,'Visible','on');
    bgcolor = get(gcf,'Color');
    set(gca,'YColor',[1 1 1]);
    set(gca,'XTick',[-2 0 1]);
    set(gca,'XTickLabel',{'-2'; 'MO';'1'},'FontSize',paper_font_size-1);  
    set(gca,'TickLength',[0.03 0.025])
    hold on;
    xlabel('Time (s)', 'FontSize',paper_font_size-1);
    
    % Annotate line
    axes(T_plot(15));
    axes_pos = get(gca,'Position'); %[lower bottom width height]
    axes_ylim = get(gca,'Ylim');
    annotate_length = (5*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
    annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.025) (axes_pos(1)+axes_pos(3)+0.025)],...
        [(axes_pos(2)+axes_pos(4) - annotate_length/5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/5)],'LineWidth',2);
    
    %hylabel = ylabel('EEG (\muV)','FontSize', 10, 'rotation',90);
    %pos = get(hylabel,'Position');
    
    %mtit(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'fontsize',14,'color',[0 0 0],'xoff',-.02,'yoff',.025);
    %mtit('LSGR, Left hand, Triggered Mode, Day 1','fontsize',14,'yoff',0.025);
    %title(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)));
    %legend('Average RP','Movement Onset','Orientation','Horizontal');
    %export_fig MS_ses1_cond1_block80_Average '-png' '-transparent';
    
    % Expand axes to fill figure
%     fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
%     hgexport(fig,'-clipboard',style,'applystyle', true);
%     drawnow;
    
    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
         %tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.tif'];
         %fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.fig'];
        %print('-dtiff', '-r300', tiff_filename); 
        %saveas(gcf,fig_filename);
    else
        disp('Save figure aborted');
    end
    %% Plot ERPs
if plot_ERPs == 1
    paper_font_size = 10;
    figure('Position',[1050 1300 3.5*116 2.5*116]); 
    %figure('units','normalized','outerposition',[0 0 1 1])
    T_plot = tight_subplot(3,5,[0.01 0.01],[0.15 0.01],[0.1 0.1]);
    hold on;
    plot_ind4 = 1;
    
    for ind4 = 1:length(Channels_nos)
        axes(T_plot(ind4));
%         if plot_ind4 == 5                 % Commented by Nikunj on Oct 31,2013
%             plot_ind4 = plot_ind4 + 1;
%         end
        %subplot(5,5,plot_ind4);
        hold on;
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:),'k','LineWidth',1.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
        plot(move_erp_time,move_avg_channels(Channels_nos(ind4),:) - (move_SE_channels(Channels_nos(ind4),:)),'-','Color',[0 0 0],'LineWidth',0.5);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:),'r','LineWidth',2);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:)+ (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(ind4),:) - (rest_SE_channels(Channels_nos(ind4),:)),'-','Color',[1 0 0],'LineWidth',0.5);
%         
%         jbfill(move_erp_time,move_avg_channels(Channels_nos(ind4),:)+ (move_SE_channels(Channels_nos(ind4),:)),...
%            move_avg_channels(Channels_nos(ind4),:)- (move_SE_channels(Channels_nos(ind4),:)),[1 1 1],'k',0,0.3);
%         plot(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
%         jbfill(rest_erp_time,rest_avg_channels(Channels_nos(RPind),:)+ (rest_SE_channels(Channels_nos(RPind),:)),...
%            rest_avg_channels(Channels_nos(RPind),:)- (rest_SE_channels(Channels_nos(RPind),:)),'r','k',0,0.3);
       
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),rest_time, rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,preprocessed_move_epochs(Channels_nos(RPind),:),'b',erp_time,standardize_move_epochs(Channels_nos(RPind),:),'k',erp_time,rest_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        %plot(erp_time,move_avg_channels(Channels_nos(RPind),:),'r','LineWidth',2);
        text(-2,-5,[EEG.chanlocs(Channels_nos(ind4)).labels],'Color','k','FontWeight','normal','FontSize',paper_font_size-1); % ', ' num2str(Channels_nos(ind4))
        set(gca,'YDir','reverse');
        %if max(abs(move_avg_channels(Channels_nos(RPind),:))) <= 6
%          if max((move_avg_channels(Channels_nos(ind4),:)+(move_SE_channels(Channels_nos(ind4),:)))) >= 6 || ...
%                  min((move_avg_channels(Channels_nos(ind4),:)-(move_SE_channels(Channels_nos(ind4),:)))) <= -6
%             axis([move_erp_time(1) move_erp_time(end) -15 15]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(gca,'FontWeight','bold');  
%             set(gca,'YTick',[-10 0 10]);
%             set(gca,'YTickLabel',{'-10'; '0'; '10'});
%         else
            axis([-2.5 1 -6 3]);
            %axis([move_erp_time(1) 1 -15 15]);
            set(gca,'YTick',[-5 0 2]);
            set(gca,'YTickLabel',{'-5'; '0'; '+2'},'FontWeight','normal','FontSize',paper_font_size-1);
%        end
        line([0 0],[-10 10],'Color','k','LineWidth',0.5,'LineStyle','--');  
        line([-2.5 1.5],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
        plot_ind4 = plot_ind4 + 1;
        
        if ind4 == 6
            set(gca,'XColor',[1 1 1],'YColor',[1 1 1])
            set(gca,'YtickLabel',' ');
             hylab = ylabel('MRCP Grand Average','FontSize',paper_font_size-1,'Color',[0 0 0]);
             pos_hylab = get(hylab,'Position');
             set(hylab,'Position',[pos_hylab(1) pos_hylab(2) pos_hylab(3)]);
        else
            set(gca,'Visible','off');
        end
        
    %    grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
     %   set(gca,'XTick',[-2 -1 0 1]);
     %   set(gca,'XTickLabel',{'-2';'-1'; '0';'1'});  
        
          
    end

    % subplot(4,3,5);
    % topoplot([],EEG.chanlocs,'style','blank','electrodes','labels','chaninfo',EEG.chaninfo);
    %subplot(5,5,8);
   
    
    axes(T_plot(11));
    set(gca,'Visible','on');
    bgcolor = get(gcf,'Color');
    set(gca,'YColor',[1 1 1]);
    set(gca,'XTick',[-2 0 1]);
    set(gca,'XTickLabel',{'-2'; 'MO';'1'},'FontSize',paper_font_size-1);  
    set(gca,'TickLength',[0.03 0.025])
    hold on;
    xlabel('Time (s)', 'FontSize',paper_font_size-1);
    
    % Annotate line
    axes(T_plot(15));
    axes_pos = get(gca,'Position'); %[lower bottom width height]
    axes_ylim = get(gca,'Ylim');
    annotate_length = (5*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
    annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.025) (axes_pos(1)+axes_pos(3)+0.025)],...
        [(axes_pos(2)+axes_pos(4) - annotate_length/5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/5)],'LineWidth',2);
    
    %hylabel = ylabel('EEG (\muV)','FontSize', 10, 'rotation',90);
    %pos = get(hylabel,'Position');
    
    %mtit(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'fontsize',14,'color',[0 0 0],'xoff',-.02,'yoff',.025);
    %mtit('LSGR, Left hand, Triggered Mode, Day 1','fontsize',14,'yoff',0.025);
    %title(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)));
    %legend('Average RP','Movement Onset','Orientation','Horizontal');
    %export_fig MS_ses1_cond1_block80_Average '-png' '-transparent';
    
    % Expand axes to fill figure
%     fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
%     hgexport(fig,'-clipboard',style,'applystyle', true);
%     drawnow;
    
    response = input('Save figure to folder [y/n]: ','s');
    if strcmp(response,'y')
         %tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.tif'];
         %fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_MRCP_grand_average.fig'];
        %print('-dtiff', '-r300', tiff_filename); 
        %saveas(gcf,fig_filename);
    else
        disp('Save figure aborted');
    end
    