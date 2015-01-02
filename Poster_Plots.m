% Plots for my poster

posterFontSize = 18;
posterLineWidth = 2;


% %%  Movement onset using EMG 
% [no_epochs,no_datapts, no_chns] = size(up_emg_epochs);
% emg_onset = [];
% for k = 1:no_epochs
%     biceps_mvc = max(up_emg_epochs(k,:,1));
%     %triceps_mvc = max(emg_epochs.emg_move_epochs(k,:,2));
%     biceps_onset_time = emg_epochs.emg_erp_time(find(up_emg_epochs(k,:,1) >= 0.1*biceps_mvc,1));
%     %triceps_onset_time = emg_epochs.emg_erp_time(find(emg_epochs.emg_move_epochs(k,:,2) >= 0.05*triceps_mvc,1));
%     emg_onset(k) = biceps_onset_time;
% end
% 
% for j = 1:length(emg_onset)
%       onset_idx = find(Average.move_erp_time ==  round(emg_onset(j)*100) /100,1);
%       bc_up_eeg(j,:) = bc_up_move_ch_avg(j,onset_idx - (2.5*200):onset_idx+(1*200));
% end
        


%% Plots for comparing MRCPSs and sEMG for Up/Down Targets
% manually load emg_epochs, velocity_epochs, average_noncausal

% Cond1 : Only UP targets at different levels of sEMG MVC
    % 1. Apply baseline correction
          emg_baseline = emg_epochs.emg_move_epochs(:,find(emg_epochs.emg_erp_time == -2.5):...
        find(emg_epochs.emg_erp_time == -2.25),1:2);
          
          bc_emg_epochs(:,:,1) = emg_epochs.emg_move_epochs(:,:,1) -...
        repmat(mean(emg_baseline(:,:,1),2),1,size(emg_epochs.emg_move_epochs,2));
      
          bc_emg_epochs(:,:,2) = emg_epochs.emg_move_epochs(:,:,2) -...
        repmat(mean(emg_baseline(:,:,2),2),1,size(emg_epochs.emg_move_epochs,2));  
          
    % 2. Find emg epochs with Up / Down Targets
        up_emg_epochs =  bc_emg_epochs(find(Velocity_Epoch(:,end)==3)-1,:,:);
        down_emg_epochs =  bc_emg_epochs(find(Velocity_Epoch(:,end)==1)-1,:,:);
        
    % 3. Find Max Voluntary Contraction sEMG value and spilt in corr epochs
        up_mvc_emg = max(up_emg_epochs(:,:,1),[],2);
        high_up_emg = up_emg_epochs(up_mvc_emg >= 0.6*max(up_mvc_emg),:,:);
        low_up_emg = up_emg_epochs(up_mvc_emg < 0.6*max(up_mvc_emg),:,:);
        
        down_mvc_emg = max(down_emg_epochs(:,:,2),[],2);
        high_down_emg = down_emg_epochs(down_mvc_emg >= 0.5*max(down_mvc_emg),:,:);
        low_down_emg = down_emg_epochs(down_mvc_emg < 0.5*max(down_mvc_emg),:,:);
        

    % 4.----------EEG Related
        move_epochs = Average.move_epochs;
        rest_epochs = Average.rest_epochs;
        classchannels = Performance.classchannels;
        move_erp_time = Average.move_erp_time;
        rest_erp_time = Average.rest_erp_time;
        move_ch_avg = mean(move_epochs(:,:,classchannels),3);
        %move_ch_avg = move_epochs(:,:,14);
        %rest_ch_avg = mean(rest_epochs(:,:,classchannels),3);
        
        up_move_ch_avg = move_ch_avg(find(Velocity_Epoch(:,end)==3)-1,:);
        %up_rest_ch_avg = rest_ch_avg(find(Velocity_Epoch(:,end)==3)-1,:);
        down_move_ch_avg = move_ch_avg(find(Velocity_Epoch(:,end)==1)-1,:);
        
        up_move_eeg_baseline = up_move_ch_avg(:,find(move_erp_time == -2.5):find(move_erp_time == -2.25));
        %rest_eeg_baseline = up_rest_ch_avg(:,find(rest_erp_time == -2.5):find(rest_erp_time == -2.25));
        bc_up_move_ch_avg = up_move_ch_avg - repmat(mean(up_move_eeg_baseline,2),1,size(up_move_ch_avg,2));
        %bc_up_rest_ch_avg = up_rest_ch_avg - repmat(mean(rest_eeg_baseline,2),1,size(up_rest_ch_avg,2));
        down_move_eeg_baseline = down_move_ch_avg(:,find(move_erp_time == -2.5):find(move_erp_time == -2.25));
        bc_down_move_ch_avg = down_move_ch_avg - repmat(mean(down_move_eeg_baseline,2),1,size(up_move_ch_avg,2));
        
        % Separate EEG epochs into high/low force movements
        high_up_eeg = bc_up_move_ch_avg(up_mvc_emg >= 0.6*max(up_mvc_emg),:);
        low_up_eeg = bc_up_move_ch_avg(up_mvc_emg < 0.6*max(up_mvc_emg),:);
        high_down_eeg = bc_down_move_ch_avg(down_mvc_emg >= 0.5*max(down_mvc_emg),:);
        low_down_eeg = bc_down_move_ch_avg(down_mvc_emg < 0.5*max(down_mvc_emg),:);
        
        
%%
        figure; 
        subplot(2,1,1)
        grid on; hold on
        plot(emg_epochs.emg_erp_time,up_emg_epochs(:,:,1),'b');
        xlim([-2.5 1]);
        ylim([0 30]);
        title('Up-Biceps');
        %subplot(2,2,2)
        grid on;
        plot(emg_epochs.emg_erp_time,down_emg_epochs(:,:,2),'r');
        xlim([-2.5 1]);
        ylim([0 30]);
        title('Down-Triceps');
        subplot(2,1,2)
        grid on;
        plot(move_erp_time,bc_up_move_ch_avg,'b');
        hold on;
        plot(move_erp_time,mean(bc_up_move_ch_avg),'k','LineWidth',2);
        title('Up-EEG');
        xlim([-2.5 1]);
        %subplot(2,2,4)
        grid on;
        hold on;
        plot(move_erp_time,bc_down_move_ch_avg,'r');
        hold on;
        plot(move_erp_time,mean(bc_down_move_ch_avg),'--k','LineWidth',2);
        title('Down-EEG');
        xlim([-2.5 1]);
        
%%
        figure; 
        subplot(2,2,1)
        grid on; hold on;
        plot(emg_epochs.emg_erp_time,high_up_emg(:,:,1),'b');
        plot(emg_epochs.emg_erp_time,low_up_emg(:,:,1),'r');
        xlim([-2.5 1]);
        ylim([0 30]);
        title('Up-Biceps');
        subplot(2,2,2)
        grid on; hold on;
        plot(emg_epochs.emg_erp_time,high_down_emg(:,:,2),'b');
        plot(emg_epochs.emg_erp_time,low_down_emg(:,:,2),'r');
        xlim([-2.5 1]);
        ylim([0 30]);
        title('Down-Triceps');
        subplot(2,2,3)
        grid on; hold on;
        plot(move_erp_time,high_up_eeg,'b');
        plot(move_erp_time,mean(high_up_eeg),'k','LineWidth',2);
        plot(move_erp_time,low_up_eeg,'r');
        plot(move_erp_time,mean(low_up_eeg),'--k','LineWidth',2);
        title('Up-EEG');
        xlim([-2.5 1]);
        
        subplot(2,2,4)
        grid on;hold on
        plot(move_erp_time,high_down_eeg,'b');
        plot(move_erp_time,mean(high_down_eeg),'k','LineWidth',2);
        plot(move_erp_time,low_down_eeg,'r');
        plot(move_erp_time,mean(low_down_eeg),'--k','LineWidth',2);
        title('Down-EEG');
        xlim([-2.5 1]);
        
%% 
% load emg_epochs, velocity_epochs, average_noncausal

        trials_from = 15;
        trials_to = 21;
        trials_to_plot = [1 5 7 9 14 19]; %trials_from:trials_to;
        %mycolors = {'r' 'k' 'b' 'g' 'c' 'm' 'y'};
        
 % 1. Apply baseline correction
          emg_baseline = emg_epochs.emg_move_epochs(:,find(emg_epochs.emg_erp_time == -2.5):...
        find(emg_epochs.emg_erp_time == -2.25),1:2);
          
          bc_emg_epochs(:,:,1) = emg_epochs.emg_move_epochs(:,:,1) -...
        repmat(mean(emg_baseline(:,:,1),2),1,size(emg_epochs.emg_move_epochs,2));
      
          bc_emg_epochs(:,:,2) = emg_epochs.emg_move_epochs(:,:,2) -...
        repmat(mean(emg_baseline(:,:,2),2),1,size(emg_epochs.emg_move_epochs,2));  
          
        % 2. 
          % Up Targets
          up_emg_epochs =  bc_emg_epochs(find(Velocity_Epoch(:,end)==3)-1,:,:);
          
        figure('Position',[50 300 10*116 4*116]);
        first_plot = tight_subplot(1,2,0.15,[0.1 0.15],[0.1 0.05]);
        axes(first_plot(1));
        %mycolors = {'-r' '-.k' '-r' '-r' '-r' '-r' '-r'};
        hold on; grid off;
        plot(emg_epochs.emg_erp_time,up_emg_epochs(trials_to_plot(1),:,1),'-.k','LineWidth',posterLineWidth);
        for k = 2:length(trials_to_plot)
            plot(emg_epochs.emg_erp_time,up_emg_epochs(trials_to_plot(k),:,1),'Color',[0.4 0.4 0.4],'LineWidth',posterLineWidth);                
        end
        ylim([0 30]);
        ylimits = ylim;
        xlim([-2.5 1]);
        %title('Biceps Activity (mV)','FontSize',12)
%         lh = legend('Biceps Activity','Location','NorthWest');
%         set(lh,'Box', 'off','Color','none');
        set(gca,'XTick',[-2 -1 0 1]);
        set(gca,'XTickLabel',{'-2' '-1' '0' '1'},'FontSize',posterFontSize);
        set(gca,'YTick',[0 ylimits(2)]);
        set(gca,'YTickLabel',{'0' num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
        
        %----------EEG Related
        move_epochs = Average.move_epochs;
        rest_epochs = Average.rest_epochs;
        classchannels = Average.RP_chans(1:4);
        move_erp_time = Average.move_erp_time;
        rest_erp_time = Average.rest_erp_time;
        move_ch_avg = mean(move_epochs(:,:,classchannels),3);
        rest_ch_avg = mean(rest_epochs(:,:,classchannels),3);
        
        up_move_ch_avg = move_ch_avg(find(Velocity_Epoch(:,end)==3)-1,:);
        up_rest_ch_avg = rest_ch_avg(find(Velocity_Epoch(:,end)==3)-1,:);
        
        move_eeg_baseline = up_move_ch_avg(:,find(move_erp_time == -2.5):find(move_erp_time == -2.25));
        rest_eeg_baseline = up_rest_ch_avg(:,find(rest_erp_time == -2.5):find(rest_erp_time == -2.25));
        bc_up_move_ch_avg = up_move_ch_avg - repmat(mean(move_eeg_baseline,2),1,size(up_move_ch_avg,2));
        bc_up_rest_ch_avg = up_rest_ch_avg - repmat(mean(rest_eeg_baseline,2),1,size(up_rest_ch_avg,2));
          
        
        mycolors = {'-b' '-.k' '-b' '-b' '-b' '-b' '-b'};
        axes(first_plot(2));
        hold on; grid off;
        for k = 1:length(trials_to_plot)
            plot(move_erp_time,up_move_ch_avg(trials_to_plot(k),:),mycolors{mod(k,7)+1},'LineWidth',posterLineWidth);                
        end
        set(gca,'YDir','reverse');
        ylim([-15 5]);
        ylimits = ylim;
        xlim([-2.5 1]);
        set(gca,'XTick',[-2 -1 0 1]);
        set(gca,'XTickLabel',{'-2' '-1' '0' '1'},'FontSize',posterFontSize);
        set(gca,'YTick',[ylimits(1) ylimits(2)]);
        set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
        %export_fig 'emg_eeg_comparison' '-png' '-transparent'
        
        
%% second plot

        figure('Position',[50 300 10*116 4*116]);
        first_plot = tight_subplot(1,2,0.05,[0.1 0.15],[0.05 0.05]);
        axes(first_plot(1));
        mycolors = {'-m' '-m' '-m' '-m' '-m' '-m' '-m'};
        hold on; grid off;
        for k = 1:length(trials_to_plot)
            plot(rest_erp_time,up_rest_ch_avg(trials_to_plot(k),:),mycolors{mod(k,7)+1},'LineWidth',posterLineWidth);                
        end
        set(gca,'YDir','reverse');
        ylim([-15 5]);
        ylimits = ylim;
        xlim([-2.5 1]);
        shd1 = find(rest_erp_time == -0.5-0.95,1,'first');
        shd2 = find(rest_erp_time == -0.5,1,'first');
        jbfill(rest_erp_time(shd1:shd2),repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'m','k',0,0.3);
        
        set(gca,'XTick',[-2 -1 -0.5 0]);
        set(gca,'XTickLabel',{'-2*' '-1*' '-0.5*' '0*'},'FontSize',posterFontSize);
        set(gca,'YTick',[ylimits(1) ylimits(2)]);
        set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
                      
        mycolors = {'-b' '-.k' '-b' '-b' '-b' '-b' '-b'};
        axes(first_plot(2));
        hold on; grid off;
        for k = 1:length(trials_to_plot)
            plot(move_erp_time,up_move_ch_avg(trials_to_plot(k),:),mycolors{mod(k,7)+1},'LineWidth',posterLineWidth);                
        end
        set(gca,'YDir','reverse');
        ylim([-15 5]);
        ylimits = ylim;
        xlim([-2.5 1]);
        shd1 = find(move_erp_time == -0.3-0.95,1,'first');
        shd2 = find(move_erp_time == -0.3,1,'first');
        jbfill(move_erp_time(shd1:shd2),repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'b','k',0,0.3);
        
        set(gca,'XTick',[-2 -1 -0.3 0]);
        set(gca,'XTickLabel',{'-2' '-1' '-0.3' '0'},'FontSize',posterFontSize);
        set(gca,'YTick',[ylimits(1) ylimits(2)]);
        %set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
        %export_fig 'conventional_feature_extraction' '-png' '-transparent'
        
%% Third plot

        figure('Position',[50 300 10*116 4*116]);
        first_plot = tight_subplot(1,2,0.05,[0.1 0.15],[0.05 0.05]);
        axes(first_plot(1));
        mycolors = {'-m' '-m' '-m' '-m' '-m' '-m' '-m'};
        hold on; grid off;
        for k = 1:length(trials_to_plot)
            plot(rest_erp_time,up_rest_ch_avg(trials_to_plot(k),:),mycolors{mod(k,7)+1},'LineWidth',posterLineWidth);                
        end
        set(gca,'YDir','reverse');
        ylim([-15 5]);
        ylimits = ylim;
        xlim([-2.5 1]);
        shd1 = find(rest_erp_time == -0.5-0.6,1,'first');
        shd2 = find(rest_erp_time == -0.5,1,'first');
        jbfill(rest_erp_time(shd1:shd2),repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'r','k',0,0.3);
        
        %title('Biceps Activity (mV)','FontSize',12)
%         lh = legend('Biceps Activity','Location','NorthWest');
%         set(lh,'Box', 'off','Color','none');
        set(gca,'XTick',[-2 -1 -0.5 0]);
        set(gca,'XTickLabel',{'-2*' '-1*' '-0.5*' '0*'},'FontSize',posterFontSize);
        set(gca,'YTick',[ylimits(1) ylimits(2)]);
        set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
                      
        mycolors = {'-b' '-.k' '-b' '-b' '-b' '-b' '-b'};
        axes(first_plot(2));
        hold on; grid off;
        mt1 = find(move_erp_time == -2,1,'first');
        mt2 = find(move_erp_time == 0,1,'first');
        mt3 = find(move_erp_time == -1.5,1,'first');
        [min_avg(:,1),min_avg(:,2)] = min(up_move_ch_avg(trials_to_plot,mt1:mt2),[],2); % value, indices

        for k = 1:length(trials_to_plot)
            plot(move_erp_time,up_move_ch_avg(trials_to_plot(k),:),mycolors{mod(k,7)+1},'LineWidth',posterLineWidth);                
            start_ind = mt1+min_avg(k,2);
            stop_ind = mt1+min_avg(k,2)-120;
            leg1 = plot(move_erp_time(start_ind),up_move_ch_avg(trials_to_plot(k),start_ind),'^r','LineWidth',posterLineWidth,'MarkerSize',8);
            if start_ind > mt3
                leg2 = plot(move_erp_time(stop_ind:start_ind),up_move_ch_avg(trials_to_plot(k),stop_ind:start_ind),'-r','LineWidth',posterLineWidth);
            else
                leg3 = plot(move_erp_time(start_ind),up_move_ch_avg(trials_to_plot(k),start_ind),'^k','LineWidth',posterLineWidth,'MarkerSize',8);
            end
        end
        set(gca,'YDir','reverse');
        ylim([-15 5]);
        ylimits = ylim;
        xlim([-2.5 1]);
        %shd1 = find(move_erp_time == -0.31-0.75,1,'first');
        %shd2 = find(move_erp_time == -0.31,1,'first');
        %jbfill(move_erp_time(shd1:shd2),repmat(ylimits(2),1,shd2-shd1+1),repmat(ylimits(1),1,shd2-shd1+1),'b','k',0,0.3);
        
        
        
        set(gca,'XTick',[-2 -1.5 -1 0 1]);
        set(gca,'XTickLabel',{'-2' '-1.5' '-1' '0' '1'},'FontSize',posterFontSize);
        set(gca,'YTick',[ylimits(1) ylimits(2)]);
        %set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);
        legend([leg1 leg2],'Negative Peaks','Optimal ''Go'' period','Location','NorthWest');                    
        legend('boxoff');
        %'Rejects trial with peak \leq -1.5s',
        %set(leg1,'Color',[1 1 1],'FontSize',posterFontSize);        % 'Color','none'
        %set(leg2,'Color',[1 1 1],'FontSize',posterFontSize);        % 'Color','none'
               
        %export_fig 'smart_feature_extraction' '-png' '-transparent'
        
%% Fourth plot - Online Simulation
   figure('Position',[50 300 3*116 3*116]);
   hold on; 
   plot(move_erp_time,up_move_ch_avg(trials_to_plot(2),:),'b','LineWidth',posterLineWidth);
   set(gca,'YDir','reverse');
    ylim([-15 5]);
    ylimits = ylim;
    xlim([-2.5 1]);
    set(gca,'XTick',[-2 -1 0 1]);
    set(gca,'XTickLabel',{'-2' '-1' '0' '1'},'FontSize',posterFontSize);
    set(gca,'YTick',[ylimits(1) ylimits(2)]);
    set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
    v = axis;
    line([0 0],[v(3) v(4)],'Color','k','LineWidth',posterLineWidth);

    export_fig 'online_simulation' '-png' '-transparent'
%% Fifth plot - ROC plot, MS_ses1_cond3

   figure('Position',[50 30 6*116 5*116]);
   fifth_plot = tight_subplot(1,2,0.1,[0.1 0.2],[0.1 0.05]);
      
   %if use_conventional_features == 1
       use_feature = 1;
       axes(fifth_plot(1)); hold on; grid off;
       conv_fea_legend = plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','-b','LineWidth',posterLineWidth);
       plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','sb','MarkerSize',3,'LineWidth',posterLineWidth,'MarkerFaceColor',[0 0 1]);
       ylim([0.3 1]);
       %ylabel('AUC','FontSize',10);
       %xlabel('Window Length (sec.)','FontSize',10);
       conv_optimal_window_length = 0.95; %input('Enter Optimal Window Length (Conventional Features): ');
       conv_opt_wl_ind = find(window_length_range./Fs_eeg == conv_optimal_window_length);
       h1 = plot(conv_optimal_window_length, roc_OPT_AUC(conv_opt_wl_ind,3,use_feature)','pk','MarkerSize',10,'LineWidth',posterLineWidth);

       axes(fifth_plot(2)); hold on; grid off;
       %plot(roc_measures(:,2), roc_measures(:,1),'xk');
       %plot3(roc_measures(:,2), roc_measures(:,1),window_length_range','xk','MarkerSize',10,'LineWidth',2);
       h2 = plot(smooth(roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,1,use_feature),10),smooth(roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,2,use_feature),2),'b','LineWidth',posterLineWidth);
       %plot(roc_measures(conv_opt_wl_ind,2,use_feature), roc_measures(conv_opt_wl_ind,1,use_feature),'xb','MarkerSize',10,'LineWidth',2);
       %text(roc_measures(conv_opt_wl_ind,2), roc_measures(conv_opt_wl_ind,1)-0.1,'p = 0.5','FontWeight','bold');

       %plot(roc_OPT_AUC(conv_opt_wl_ind,1,use_feature), roc_OPT_AUC(conv_opt_wl_ind,2,use_feature),'xk','MarkerSize',10,'LineWidth',2);
       %p_opt_ind = find(roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(:,2,use_feature) == roc_OPT_AUC(conv_opt_wl_ind,2,use_feature),1,'first');
       %text(roc_OPT_AUC(conv_opt_wl_ind,1,use_feature), roc_OPT_AUC(conv_opt_wl_ind,2,use_feature)+0.1,sprintf('p(opt) = %.2f',roc_X_Y_Thr{conv_opt_wl_ind,use_feature}(p_opt_ind,3,use_feature)),'FontWeight','bold'); % Change here           
       axis([0 1 0 1]);
       %xlabel('FPR','FontSize',10);
       %ylabel('TPR','FontSize',10);
       %title('ROC Curve','FontSize',12);
       %legend(h1,'ROC Curve','Location','SouthEast');
       xlim([0 1]);
   %end 
   %if use_smart_features == 1
       use_feature = 2;
       axes(fifth_plot(1)); hold on; grid off;
       smart_fea_legend = plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','-r','LineWidth',posterLineWidth);
       plot(window_length_range./Fs_eeg, roc_OPT_AUC(:,3,use_feature)','sr','MarkerSize',3,'LineWidth',posterLineWidth,'MarkerFaceColor',[0 0 1]);
       ylim([0.3 1]);
       xlim([0 1]);
       %ylabel('AUC','FontSize',10);
       %xlabel('Window Length (sec.)','FontSize',10);
       smart_optimal_window_length = 0.6;%input('Enter Optimal Window Length (Smart Features): ');
       smart_opt_wl_ind = find(window_length_range./Fs_eeg == smart_optimal_window_length);
       plot(smart_optimal_window_length, roc_OPT_AUC(smart_opt_wl_ind,3,use_feature)','pk','MarkerSize',10,'LineWidth',posterLineWidth);
       set(gca,'XTick',[0 0.5 1]);
        set(gca,'XTickLabel',{'0' '0.5' '1'},'FontSize',posterFontSize);
        set(gca,'YTick',[0 1]);
        set(gca,'YTickLabel',{'0' '1'},'FontSize',posterFontSize);

       axes(fifth_plot(2)); hold on; grid off;
       %plot(roc_measures(:,2), roc_measures(:,1),'xk');
       %plot3(roc_measures(:,2), roc_measures(:,1),window_length_range','xk','MarkerSize',10,'LineWidth',2);
       h3 = plot(smooth(roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,1,use_feature),10),smooth(roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,2,use_feature),10),'r','LineWidth',posterLineWidth);
       %plot(roc_measures(smart_opt_wl_ind,2,use_feature), roc_measures(smart_opt_wl_ind,1,use_feature),'xr','MarkerSize',10,'LineWidth',2);
       %text(roc_measures(smart_opt_wl_ind,2), roc_measures(smart_opt_wl_ind,1)-0.1,'p = 0.5','FontWeight','bold');

       %plot(roc_OPT_AUC(smart_opt_wl_ind,1,use_feature), roc_OPT_AUC(smart_opt_wl_ind,2,use_feature),'xk','MarkerSize',10,'LineWidth',2);
       %p_opt_ind = find(roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(:,2,use_feature) == roc_OPT_AUC(smart_opt_wl_ind,2,use_feature),1,'first');
       %text(roc_OPT_AUC(smart_opt_wl_ind,1,use_feature), roc_OPT_AUC(smart_opt_wl_ind,2,use_feature)+0.1,sprintf('p(opt) = %.2f',roc_X_Y_Thr{smart_opt_wl_ind,use_feature}(p_opt_ind,3,use_feature)),'FontWeight','bold'); % Change here           
       axis([0 1 0 1]);
       %xlabel('FPR','FontSize',10);
       %ylabel('TPR','FontSize',10);
       %title('ROC Curve','FontSize',12);
       %legend(h1,'ROC Curve','Location','SouthEast');
       xlim([0 1]);
       line([0 1],[0 1],'Color','k','LineWidth',posterLineWidth-0.5,'LineStyle','--');
       set(gca,'XTick',[0 1]);
       set(gca,'XTickLabel',{'0' '1'},'FontSize',posterFontSize);
       set(gca,'YTick',[0 1]);
       set(gca,'YTickLabel',{'0' '1'},'FontSize',posterFontSize);
       legend([h1 h2 h3],'Optimal period','fixed ''Go'' window','flexible ''Go'' window','Location','SouthOutside','Orientation','Horizontal');                    
       %legend('boxoff'); 
   %export_fig 'area_roc_curves' '-png' '-transparent'

%% Sixth Plot - Accuracy for all subjects

figure('Position',[50 300 5*116 3*116]);
hold on;
Subject_name = {'MR','TA','CR','MS'};
subj_colors = {'g','b','r','k'};
sess_param = [ 1; % TA
               1; % CR
               1; % MR
               1]; % MS          
for subj_n = 1:4
    
        Cond_num = 3;
        Sess_num = 1;
        Block_num = 80;
        
        folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
        load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_smart_online_nic.mat']);
        online_fixed(subj_n,:) = Performance.All_eeg_accur{Performance.conv_opt_wl_ind}(1,:);
        online_variable(subj_n,:) = Performance.All_eeg_accur{Performance.smart_opt_wl_ind}(2,:);
        plot(1,mean(online_fixed(subj_n,:)),'s','Color',subj_colors{subj_n},'MarkerSize',10,'MarkerFaceColor',subj_colors{subj_n}); hold on;
        errorbar(1,mean(online_fixed(subj_n,:)),std(online_fixed(subj_n,:)),'Color',subj_colors{subj_n},'LineWidth',posterLineWidth-0.5);
        plot(3,mean(online_variable(subj_n,:)),'s','Color',subj_colors{subj_n},'MarkerSize',10,'MarkerFaceColor',subj_colors{subj_n});
        errorbar(3,mean(online_variable(subj_n,:)),std(online_variable(subj_n,:)),'Color',subj_colors{subj_n},'LineWidth',posterLineWidth-0.5);
        h(subj_n)= line([1 3],[mean(online_fixed(subj_n,:)) mean(online_variable(subj_n,:))],'Color',subj_colors{subj_n},'LineWidth',posterLineWidth);        
end
xlim([0 4]);
ylim([40 100]);
ylimits = ylim;
set(gca,'XTick',[1 3]);
set(gca,'XTickLabel',{'' ''},'FontSize',posterFontSize);
set(gca,'YTick',[ylimits(1) ylimits(2)]);
set(gca,'YTickLabel',{num2str(ylimits(1)) num2str(ylimits(2))},'FontSize',posterFontSize);
legend([h(2) h(1) h(3) h(4)],'Healthy 1','Healthy 2','Healthy 3','Stroke','Location','EastOutside','Orientation','Vertical');
%export_fig 'accuracy_all_subjects1' '-png' '-transparent'

%% Seventh Figure - Comparison across studies

Subject_name = {'MR','TA','CR','MS'};
subj_colors = {'g','b','r','k'};
sess_param = [ 1; % TA
               1; % CR
               1; % MR
               1]; % MS          
for subj_n = 1:4
    
        Cond_num = 3;
        Sess_num = 1;
        Block_num = 80;
        
        folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
        load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_smart_online_nic.mat']);
        TPR_subject_mean(subj_n,:) = mean(Performance.All_eeg_sensitivity{Performance.smart_opt_wl_ind}(2,:));
        TPR_subject_std(subj_n,:) = std(Performance.All_eeg_sensitivity{Performance.smart_opt_wl_ind}(2,:));
                
end

study_TPRs = [4 mean(TPR_subject_mean) mean(TPR_subject_std); % bhagat2014
              6 0.81 0.1;                                     % Jochumsen2013                                
              24 0.76 0.07];                                    % Lew2012    

%Xu2014 = [1440 0.79 0.11]; 
figure('Position',[50 300 5*116 3*116]);
[hbar,hbarerr] = barwitherr(study_TPRs(:,3),study_TPRs(:,2));
set(hbarerr,'LineWidth',posterLineWidth);
set(hbar,'LineWidth',posterLineWidth);
set(hbar,'BarWidth',0.5);
set(hbar,'FaceColor',1/255*[148 0 230]);
ylim([0 1.2]);
xlim([3 12]);
set(gca,'FontSize',posterFontSize);
set(gca,'YTick',[0 1]);
set(gca,'YTickLabel',{'0' '1'});
%ylabel('Average Classification Accuracy (%)','FontSize',myfont_size);
set(gca,'XTick',[4 6 10]);
set(gca,'XTickLabel',{'4' '6' '24'});
%export_fig 'study_comparison' '-png' '-transparent'



