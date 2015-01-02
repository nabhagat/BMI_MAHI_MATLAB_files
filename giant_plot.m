% Raster plots of MRCP across subjects under each condition. EMBS2014
% By Nikunj Bhagat, Graduate Student, University of Houston
% - 3/13/14
clear;
              
myfont_size = 10;
myline_width = 2.0;
myfont_weight = 'bold';
move_window = [-3.5 1.0];
baseline_int = [-3.5 -3.25];

mychnames = {'FCz','Cz','FC1','FC2','C1','C2','C3','C4'};
mychannels = [32    14    9     10   48   49   13   15];
mycolors = {'-b','-r','-k','-.k','-m','-.m','-.r','-.b'}; %[0.2 0.2 0.2], [0.4 0.4 0.4], [0.8 0.8 0.8], [0.6 0.6 0.6]};

RP_plot = 0;
Accuracy_plot = 1;
emg_velocity_plot = 0;
calc_statistics = 0;
smart_features_accuracy = 0;

%% RP_plot
if RP_plot == 1
    
% Subject Details
Subject_name = {'MR', 'MS'};
sess_param = [ %1 1 1 1; % TA
               %1 1 1 1; % CR
               2 2 1 1; % MR
               1 2 1 2]; % MS
           
%figure; 
% get(0);
figure('Position',[5 0 567 580])
T_plot = tight_subplot(5,2,[0.005 0.005],[0.1 0.15],[0.13 0.1]);
%T_plot = tight_subplot(5,2,[0.005 0.005],[0.2 0.15],[0.3 0.3]);
raster_chns = [%14 32 9 48;    % TA
                32 9 10 14;   % MR 
               14 32 10 49];  % MS  

for subj_n = 1:2
        
    Sess_num = sess_param(subj_n,1);
    Cond_num = 1;
    folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
    load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_emg_epochs.mat']);
    load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_velocity_epochs.mat']);
    load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_embs.mat']);
        % Plot EMG traces
        
        % 1. Apply baseline correction
        bc_emg_epochs = [];
          emg_baseline = emg_epochs.emg_move_epochs(:,find(emg_epochs.emg_erp_time == -3.5):...
        find(emg_epochs.emg_erp_time == -3.25),1:2);
          bc_emg_epochs(:,:,1) = emg_epochs.emg_move_epochs(:,:,1) -...
        repmat(mean(emg_baseline(:,:,1),2),1,size(emg_epochs.emg_move_epochs,2));
          bc_emg_epochs(:,:,2) = emg_epochs.emg_move_epochs(:,:,2) -...
        repmat(mean(emg_baseline(:,:,2),2),1,size(emg_epochs.emg_move_epochs,2));  
          
        % 2. Up & Down Targets
          up_emg_traces =  squeeze(median(bc_emg_epochs(find(Velocity_Epoch(:,end)==3)-1,:,1:2),1));
          %up_emg_traces = zscore(up_emg_traces);
          up_emg_baseline = mean(up_emg_traces(find(emg_epochs.emg_erp_time == -3.5):find(emg_epochs.emg_erp_time == -3.25),:));
          up_emg_traces = up_emg_traces - repmat(up_emg_baseline,size(up_emg_traces,1),1);
          %figure; plot(up_emg_traces);
          
          down_emg_traces = squeeze(median(bc_emg_epochs(find(Velocity_Epoch(:,end)==1)-1,:,:),1));  
          %up_emg_traces = zscore(up_emg_traces);
          down_emg_baseline = mean(down_emg_traces(find(emg_epochs.emg_erp_time == -3.5):find(emg_epochs.emg_erp_time == -3.25),:));
          down_emg_traces = down_emg_traces - repmat(down_emg_baseline,size(down_emg_traces,1),1);
                            
          axes(T_plot(subj_n)); hold on; grid off;
          up_handle = plot(emg_epochs.emg_erp_time,up_emg_traces(:,1),'-','Color',[0.4 0.4 0.4],'LineWidth',myline_width);
          %plot(emg_epochs.emg_erp_time,up_emg_traces(:,2),'-r','LineWidth',myline_width)
          %plot(emg_epochs.emg_erp_time,down_emg_traces(:,1),'-.b','LineWidth',myline_width);
          down_handle = plot(emg_epochs.emg_erp_time,down_emg_traces(:,2),'-.','Color',[0.4 0.4 0.4],'LineWidth',myline_width)
          axis([-3.5 1 0 10]);
          v = axis;
          line([0 0],[v(3) v(4)],'Color','k','LineWidth',myline_width-0.5,'LineStyle','-');
          line([Performance.move_window(1) Performance.move_window(1)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
          line([Performance.move_window(2) Performance.move_window(2)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
          line([Performance.rest_window(1) Performance.rest_window(1)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
          line([Performance.rest_window(2) Performance.rest_window(2)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
          text(Performance.move_window(1) + 0.2, v(4)+1.5,'{Go}','FontSize',8);
          %text(Performance.move_window(1), v(4)-1,[num2str(Performance.window_length*1E3) ' ms'],'FontSize',11);
          text(Performance.rest_window(1) + 0.0, v(4)+1.5,'{No-Go}','FontSize',8);
          
          
          if Cond_num == 1
            if strcmp(Subject_name{subj_n},'MS')
                %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
                %title('S1','FontSize',myfont_size,'FontWeight','normal','FontAngle','italic');
                text(-1.75, v(4) + 4, 'S1','FontSize',myfont_size,'FontWeight','normal','FontAngle','italic');
            else
                %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
                %title('H1','FontSize',myfont_size,'FontWeight','normal','FontAngle','italic');
                text(-2.5, v(4) + 4, 'Subject H3','FontSize',myfont_size,'FontWeight','normal','FontAngle','italic');
            end
          end
          if subj_n == 1
                ylabel('EMG (AU)','FontSize',myfont_size,'FontWeight','normal');
                emg_leg = legend([up_handle down_handle],'Target Up','Target Down',...
                'Location','NorthWest','Orientation','Horizontal');                    
              set(emg_leg,'Color',[1 1 1],'FontSize',myfont_size);        % 'Color','none'
              htitle = get(emg_leg,'Title');
              set(htitle,'String','Muscle Activity','FontSize',myfont_size)
          end
    for num_cond = 1:4
        Cond_num = num_cond;
        Sess_num = sess_param(subj_n,Cond_num);
        Block_num = 80;
        
        folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
        load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_average_noncausal.mat']);
        load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_velocity_epochs.mat']);
                  
        Fs_eeg = Average.Fs_eeg;
        [no_epochs,no_datapts,no_channels] = size(Average.move_epochs);
        move_erp_time = Average.move_erp_time;
        
        mlim1 = round(abs( move_erp_time(1)-(move_window(1)))*Fs_eeg+1);
        mlim2 = round(abs( move_erp_time(1)-(move_window(2)))*Fs_eeg+1);

        % Apply Baseline Correction
        for epoch_cnt = 1:no_epochs
            for channel_cnt = 1:no_channels
                move_mean_baseline(channel_cnt,epoch_cnt) = mean(Average.move_epochs(epoch_cnt,...
                        find(move_erp_time == baseline_int(1)):find(move_erp_time == baseline_int(2)),channel_cnt));
                bc_move_epochs(epoch_cnt,:,channel_cnt) = Average.move_epochs(epoch_cnt,:,channel_cnt)...
                    - move_mean_baseline(channel_cnt,epoch_cnt);
            end
        end
        
        raster_move_avg_channels = squeeze(mean(bc_move_epochs,1))';
        average_eeg = raster_move_avg_channels(raster_chns(subj_n,:),mlim1:mlim2);
        %SE_eeg = 1.96.*(Average.move_std_channels./sqrt(no_epochs));
        %average_velocity = mean(Velocity_Epoch);
        raster_time = move_erp_time(mlim1:mlim2);

        % Plot the rasters; Adjust parameters for plot
        [raster_row,raster_col] = size(average_eeg);
        add_offset = 0;
        axes(T_plot(subj_n + 2*(Cond_num-1) + 2)); 
        hold on;   

        set(gca,'YDir','reverse');
        for raster_index = 1:raster_row;
            if Cond_num == 1
                h(subj_n,raster_index) = plot(raster_time,add_offset + average_eeg(raster_index,:),mycolors{mychannels == raster_chns(subj_n,raster_index)},'LineWidth',myline_width);
                hold on;
                
            else
                %raster_zscore(raster_index,:) = raster_zscore(raster_index,:) + add_offset*raster_index;  % Add offset to each channel of raster plot
                plot(raster_time,add_offset + average_eeg(raster_index,:),mycolors{mychannels == raster_chns(subj_n,raster_index)},'LineWidth',myline_width);
                hold on;
            end
        end
        
        %plot(raster_time,-5*average_velocity(mlim1:mlim2)+2,'Color', [0.4 0.4 0.4],'LineWidth',2);
        axis([move_window(1) move_window(2) -6 1]);
        %set(gca,'YGrid','on');
        grid off;
        v = axis;
        %grid off;
        line([0 0],[v(3) v(4)],'Color','k','LineWidth',myline_width-0.5,'LineStyle','-');
        line([Performance.move_window(1) Performance.move_window(1)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
        line([Performance.move_window(2) Performance.move_window(2)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
        line([Performance.rest_window(1) Performance.rest_window(1)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');
        line([Performance.rest_window(2) Performance.rest_window(2)],[v(3) v(4)],'Color',[0 0 0],'LineWidth',myline_width-0.5,'LineStyle','--');

        set(gca,'YTick',[-5 0])
        set(gca,'XTick',[-3 -2 -1 0 1]);
        
        if subj_n == 1
            set(gca,'YTickLabel',{'-5' '0'},'FontSize',myfont_size,'FontWeight','normal');
            if Cond_num == 2
                ylabel('Average EEG (\muV)','FontSize',myfont_size,'FontWeight','normal');
            end
        end
        if subj_n == 2
            
            switch Cond_num
                case 1
                    text(1.25,1,'BackDrive','FontSize',myfont_size,'FontWeight','normal','Rotation',90);                   
                case 2
                    text(1.25,0,'Passive','FontSize',myfont_size,'FontWeight','normal','Rotation',90);
                case 3
                    text(1.25,0,'Triggered','FontSize',myfont_size,'FontWeight','normal','Rotation',90);
                case 4
                    text(1.25,1,'Observation','FontSize',myfont_size,'FontWeight','normal','Rotation',90);
                    
                    %xlabel('Time
                    %(sec.)','FontSize',myfont_size,'FontWeight','normal'); % Only
                    %for more than 2 subjects
            end
        end
        if Cond_num == 4
            set(gca,'XTickLabel',{'-3' '-2' '-1' '0' '1'},'FontSize',myfont_size,'FontWeight','normal');          
            xlabel('Time (sec.)','FontSize',myfont_size,'FontWeight','normal');
            if subj_n == 1
                %ylabel('Avg. EEG (\muV)','FontSize',myfont_size,'FontWeight','normal');
            end
        end
        
     end
end

% Create legends
legend_chns = raster_chns; 
legend_chns(2,1) = 0;
legend_chns(2,2) = 0;
legend_chns = legend_chns(:);

legend_hands = h(:);
legend_hands = legend_hands(legend_chns ~= 0);
plot_chns = legend_chns(legend_chns ~= 0);

leg = legend(legend_hands,...
        mychnames{mychannels == plot_chns(1)},...
        mychnames{mychannels == plot_chns(2)},...
        mychnames{mychannels == plot_chns(3)},...
        mychnames{mychannels == plot_chns(4)},...
        mychnames{mychannels == plot_chns(5)},...
        mychnames{mychannels == plot_chns(6)},...
        'Location','NorthWest','Orientation','Horizontal');                    
set(leg,'Color',[1 1 1],'FontSize',myfont_size);        % 'Color','none'
% linesInPlot = findobj('type','line');
% get(linesInPlot(2),'XData')
% set(linesInPlot(2),'XData',[0.1231 0.4])

htitle = get(leg,'Title');
set(htitle,'String','EEG channels','FontSize',myfont_size)
end
%export_fig 'RP_plot_embs_2014_H3' '-png' '-transparent'
%% Box plot
if Accuracy_plot == 1

    %% Subject Details
Subject_name = {'TA','CR','MR','MS'};
sess_param = [ 1 1 1 1; % TA
               1 1 1 1; % CR
               2 2 1 1; % MR
               1 2 1 2]; % MS
           
% figure;
% T_plot = tight_subplot(1,4,[0.005 0.005],[0.2 0.2],[0.2 0.2]);
figure('Position',[25 400 567 250])
T_plot = tight_subplot(1,4,[0.005 0.005],[0.1 0.15],[0.13 0.1]);
group_names = {'BD','P','T', 'O'};

bar_acc = [];
healthy_acc = [];
for subj_n = 1:4
    for num_cond = 1:4
        Cond_num = num_cond;
        Sess_num = sess_param(subj_n,Cond_num);
        Block_num = 80;
        
        folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
        load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_optimized_fixed_embs.mat']);
        Acc(:,Cond_num) = Performance.eeg_accur';
    end
    bar_acc = [bar_acc; [mean(Acc) std(Acc)]];
    healthy_acc = [healthy_acc; Acc];
    axes(T_plot(subj_n));
    h = boxplot(Acc,'labels',group_names,'plotstyle','traditional','widths',0.2,'labelorientation','horizontal');
    set(findobj(gca,'Type','text'),'FontSize',myfont_size,'FontWeight',myfont_weight);
%     Text_pos = get(findobj(gca,'Type','text'),'Position');
%     for j = 1:4
%         Text_pos{j,1} = Text_pos{j,1} + [0, -10, 0];
%     end
%     set(findobj(gca,'Type','text'),'Position',Text_pos);
    ylim([40 100]);
    %xlim([1 5]);
    set(h,'LineWidth',myline_width);
    set(gca,'YTick',[50 75 100]);
    set(gca,'YTickLabel','');
    %set(gca,'XTickLabel',{' '});
    %set(gca,'XTick',[1 2 3 4]);
    %set(gca,'XTickLabel',{'BD' 'P' 'T' 'O'});
    set(gca,'FontSize',myfont_size-1,'FontWeight',myfont_weight);
    set(gca,'YTick',[50 75 100]);
    if subj_n == 1
        set(gca,'YTick',[50 75 100]);
        set(gca,'YTickLabel',{'50' '75' '100'});
        ylabel('Classification Accuracy (%)','FontSize',myfont_size,'FontWeight',myfont_weight);
        %y = xlabel('Training Modes','FontSize',myfont_size);
        %set(y, 'position', get(y,'position')-[0,20,0]);  % shift the y label
    end
    
    %xlabh = get(gca,'XLabel'); 
    %set(xlabh,'Position',get(xlabh,'Position') - [0 10 0]);
    
    
    if strcmp(Subject_name{subj_n},'MS')
        %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
        title(['S1'],'FontSize',myfont_size,'FontWeight',myfont_weight,'FontAngle','italic');
    else
        %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
        %if subj_n == 1
            title(['H' num2str(subj_n)],'FontSize',myfont_size,'FontWeight',myfont_weight,'FontAngle','italic');
        %else
        %    title(num2str(subj_n),'FontSize',myfont_size,'FontWeight','normal','FontAngle','italic');
        %end
    end
    
    
end

%export_fig 'Accuracy_plot_all_subjects_embs' '-png' '-transparent'
%% Bar plot
% figure; 
% barwitherr(bar_acc(:,5:8),bar_acc(:,1:4))
% ylim([40 100])
% set(gca,'FontSize',myfont_size);
% set(gca,'YTick',[50 60 70 80 90 100]);
% set(gca,'YTickLabel',{'50' '60' '70' '80' '90' '100'});
% ylabel('Average Classification Accuracy (%)','FontSize',myfont_size);
% set(gca,'XTick',[1 2 3 4]);
% set(gca,'XTickLabel',{'1' '2' '3' '4 (stroke)'});
% xlabel('Subjects','FontSize',myfont_size);
% leg = legend('Backdrive','Passive','Triggered','Observation','Location','NorthEastOutside','Orientation','Vertical');                    
% set(leg,'Color',[1 1 1],'FontSize',myfont_size);        % 'Color','none'
% htitle = get(leg,'Title');
% set(htitle,'String','Training Modes','FontSize',myfont_size)
%export_fig 'Accuracy_bar_plot' '-png' '-transparent'
end
%% EMG-Velocity plot.
if emg_velocity_plot == 1
% targets = [];
% targets = [targets; kinematics(kin_move_trig_index,21)];

Subject_name = 'MS';
Sess_num = 1;
Cond_num = 1;
myfont_size = 14;
folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name '_Session' num2str(Sess_num) '\'];  
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_targets.mat']);

% Now, targets are added as last column to Velocity_Epochs - 4/10/14
%load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_velocity_epochs.mat']);

EMG_epochs = EEG.data(1:2,:,:);
Repochs = [];
for k = 1:80
    Repochs(:,:,k) = [downsample(EMG_epochs(:,:,k)',5)' zeros(1,2)'] ;
end

upper_traces = zscore([mean(Velocity_Epoch([logical(0);targets==3],:));mean(Repochs(:,:,targets==3),3)],0,2);
lower_traces = zscore([mean(Velocity_Epoch([logical(0);targets==1],:));mean(Repochs(:,:,targets==1),3)],0,2);

figure;
T_plot = tight_subplot(2,1,[0.01 0.01],[0.2 0.2],[0.3 0.3]);
tlim = find(Velocity_Epoch(1,:) == 2.00);

axes(T_plot(1));
hold on; set(gca,'XGrid','on');
h1 = plot(Velocity_Epoch(1,1:tlim),upper_traces(1,1:tlim),'Color',[0.5 0.5 0.5],'LineWidth',2);
h2 = plot(Velocity_Epoch(1,1:tlim),upper_traces(2,1:tlim),'b','LineWidth',2);
h3 = plot(Velocity_Epoch(1,1:tlim),upper_traces(3,1:tlim),'Color',[0.0 0.7 0],'LineWidth',2);
axis([-2.5 2 -1 3]);
line([0 0],[-5 5],'Color','k','LineWidth',2);
set(gca,'XTick',[-2 -1 0 1 2]);
% set(gca,'XTickLabel',{'-2' '-1' '0' '1' '2'},'FontSize',myfont_size,'FontWeight','normal'); 
% xlabel('Time (sec.)','FontSize',myfont_size);
set(gca,'FontSize',myfont_size);
leg1 = legend([h1 h2 h3],'Velocity (Elbow Joint)','EMG Envelop (Biceps)','EMG Envelop (Triceps)','Location','NorthEastOutside');
set(leg1,'FontSize',myfont_size);        % 'Color','none'
htitle = get(leg1,'Title');
set(htitle,'String','Average(Active Mode)','FontSize',myfont_size)
text(-2.2,2.5,'Upper Target','FontSize',myfont_size);
hold off;

axes(T_plot(2));
hold on; set(gca,'XGrid','on');
plot(Velocity_Epoch(1,1:tlim),lower_traces(1,1:tlim),'Color',[0.5 0.5 0.5],'LineWidth',2);
plot(Velocity_Epoch(1,1:tlim),lower_traces(2,1:tlim),'b','LineWidth',2);
plot(Velocity_Epoch(1,1:tlim),lower_traces(3,1:tlim),'Color',[0.0 0.7 0],'LineWidth',2);
axis([-2.5 2 -1.2 3]);
line([0 0],[-5 5],'Color','k','LineWidth',2);
set(gca,'XTick',[-2 -1 0 1 2]);
set(gca,'XTickLabel',{'-2' '-1' '0' '1' '2'},'FontSize',myfont_size,'FontWeight','normal'); 
xlabel('Time (sec.)','FontSize',myfont_size);
set(gca,'FontSize',myfont_size);
text(-2.2,2.5,'Lower Target','FontSize',myfont_size);
hold off;

%export_fig 'emg_velocity_plot' '-png' '-transparent'
end    
%% Calculate Statistics
if calc_statistics == 1
    % One-sided Wilcoxon signed rank test to determine significant difference
    % from chance level
    for j = 1:4
        [p1(j),h1(j),stats1(j).stats]=signrank(healthy_acc(1:30,j),50,'tail','right','alpha',0.0062);
        [p2(j),h2(j),stats2(j).stats]=signrank(healthy_acc(31:40,j),50,'tail','right','alpha',0.0062);
    end

    % Nonparameteric Kruskalwallis test to compare accuracies within
    % different training modes
    group1 = [ones(10,4); 2*ones(10,4);3*ones(10,4);4*ones(10,4)];
    [kp,tbl,kstats] = kruskalwallis(healthy_acc);

    % Multiple comparison test to which mode is significantly difference than
    % other modes. 
    multcompare(kstats,'alpha',0.05,'display','on') % Re-check
end
%% Online Simulation, Compare_smart_old_features
if smart_features_accuracy == 1
    
figure;
T_plot = tight_subplot(2,2,[0.05 0.05],[0.2 0.2],[0.3 0.3]);
group_names = {'fixed','variable'};

Subject_name = {'TA', 'MS'};
sess_param = [1 1 1 1;  % TA
              1 2 1 2]; % MS

for subj_n = 1:2
    for num_cond = 1:2:3
        Cond_num = num_cond;
        Sess_num = sess_param(subj_n,Cond_num);
        Block_num = 80;
        
        for kbt = 1:-1:0
            folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
            load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_new' num2str(kbt) '.mat']);
            Perf_plot = Performance.eeg_accur';
        
 
            if subj_n == 1 && Cond_num == 1
                plot_num = 1;
            elseif subj_n == 1 && Cond_num == 3
                plot_num = 3;
            elseif subj_n == 2 && Cond_num == 1
                plot_num = 2;
            elseif subj_n == 2 && Cond_num == 3
                plot_num = 4;
            end
            axes(T_plot(plot_num));hold on;
            
            
            % all_fixed all_var good_fixed good_var
            %-------fixed------------variable--------------
            %--all----------good-------all------good-------
            
            if kbt == 0
                 % good trials
                 h1 = boxplot([Perf_plot(:,1) Perf_plot(:,2)],'labels',group_names,'plotstyle','traditional','widths',0.3,'labelorientation','horizontal','positions',[1.5 3]);
                 set(gca,'XTickLabel',{' '});
                 %set(findobj(gca,'Type','text'),'FontSize',myfont_size-2,'FontWeight',myfont_weight);
                 text(1,10,'Fixed','FontSize',myfont_size,'FontWeight','normal');
                 text(2.5,10,'Variable','FontSize',myfont_size,'FontWeight','normal');
                 set(h1,'LineWidth',myline_width);
                 set(h1,'LineStyle','-');
            else
                 % all trials
                 h2 = boxplot([Perf_plot(:,1) Perf_plot(:,2)],'labels',group_names,'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','positions',[1 2.5],'colors',[0.5 0.5 0.5]);
                 set(gca,'XTickLabel',{' '});
                 set(h2,'LineWidth',myline_width);
                 set(h2,'LineStyle','-');
                 %set(findobj('Tag','Box'),'Color',[0.5 0.5 0.5]);
                 set(findobj('Tag','Median'),'Color',[1 0 0]);
            end
        
    ylim([40 100]); 
    set(gca,'YTickLabel',{' '});
    
            if kbt == 0
                if subj_n == 1
                        set(gca,'YTick',[50 75 100]);
                        set(gca,'YTickLabel',{'50' '75' '100'});
                        set(gca,'FontSize',myfont_size,'FontWeight','bold');
                        %ylabel('Classification Accuracy (%)','FontSize',myfont_size,'FontWeight',myfont_weight);
                end

                if subj_n == 2

                        switch Cond_num
                            case 1
                                text(1.25,0.5,'Backdrivable','FontSize',myfont_size+2,'FontWeight','normal','Rotation',90);                   
                            case 2
                                %text(1.25,0.5,'Passive','FontSize',myfont_size+2,'FontWeight','normal','Rotation',90);
                            case 3
                                text(1.25,0.5,'Triggerd','FontSize',myfont_size+2,'FontWeight','normal','Rotation',90);
                            case 4
                                %text(1.25,0.5,'Observation','FontSize',myfont_size+2,'FontWeight','normal','Rotation',90);

                                %xlabel('Time %(sec.)','FontSize',myfont_size,'FontWeight','normal'); % Only
                                %for more than 2 subjects
                        end
                end

                if Cond_num == 1
                    if strcmp(Subject_name{subj_n},'MS')
                        %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
                        title('S1','FontSize',myfont_size,'FontWeight','bold','FontAngle','italic');
                    else
                        %title(Subject_name{subj_n},'FontSize',11,'FontWeight','bold');
                        title('H1','FontSize',myfont_size,'FontWeight','bold','FontAngle','italic');
                    end
                end
            end
        end
    end
    
end
%export_fig 'online_simulation_accuracy' '-png' '-transparent'
end
%% Backup

% figure;
% T_plot = tight_subplot(1,2,[0.025 0.025],[0.2 0.2],[0.2 0.2]);
% group_names = {'fixed window',' ','variable window',' '};
% 
% Subject_name = {'TA'};
% sess_param = [1 1 1 1]; % TA
% 
% for subj_n = 1:1
%     for num_cond = 1:2:3
%         Cond_num = num_cond;
%         Sess_num = sess_param(subj_n,Cond_num);
%         Block_num = 80;
%         for kbt = 0:1
%             folder_path = ['F:\Nikunj_Data\Mahi_Experiment_Data\' Subject_name{subj_n} '_Session' num2str(Sess_num) '\'];  
%             load([folder_path Subject_name{subj_n} '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_performance_new' num2str(kbt) '.mat']);
%             Acc (:,:,kbt+1)= Performance.eeg_accur';
%         end
%  
%         if Cond_num == 3
%             axes(T_plot(2));
%         else
%             axes(T_plot(1));
%         end
%         % all_fixed all_var good_fixed good_var
%         
%         %-------fixed------------variable--------------
%         %--all----------good-------all------good-------
%         
%         h = boxplot([Acc(:,1,2) Acc(:,1,1) Acc(:,2,2) Acc(:,2,1)],'labels',{'all','good','all','good'},'plotstyle','traditional','widths',0.5,'labelorientation','horizontal');
%         set(findobj(gca,'Type','text'),'FontSize',myfont_size,'FontWeight',myfont_weight);
%         
%     ylim([0 100]);
%     %xlim([1 5]);
%     set(h,'LineWidth',myline_width);
%     set(gca,'YTick',[0 25 50 75 100]);
%     set(gca,'YTickLabel','');
%     %set(gca,'XTickLabel',{' '});
%     %set(gca,'XTick',[1 2 3 4]);
%     %set(gca,'XTickLabel',{'BD' 'P' 'T' 'O'});
%     set(gca,'FontSize',myfont_size-2,'FontWeight',myfont_weight);
%     line([2.5 2.5],[0 100],'LineWidth',2,'Color',[0 0 0],'LineStyle','--');
%     text(1,90,'Fixed Window','FontSize',myfont_size-2,'FontWeight',myfont_weight);
%     text(3,90,'Variable Window','FontSize',myfont_size-2,'FontWeight',myfont_weight);
%     
%         if Cond_num == 1
%             set(gca,'YTick',[0 25 50 75 100]);
%             set(gca,'YTickLabel',{'0' '25' '50' '75' '100'});
%             ylabel('Classification Accuracy (%)','FontSize',myfont_size,'FontWeight',myfont_weight);
%             %y = xlabel('Training Modes','FontSize',myfont_size);
%             %set(y, 'position', get(y,'position')-[0,20,0]);  % shift the y label
%             title('Backdrive','FontSize',myfont_size,'FontWeight',myfont_weight,'FontAngle','italic');
%         else
%             title('Triggered','FontSize',myfont_size,'FontWeight',myfont_weight,'FontAngle','italic');
%         end
% 
%     end
%     
% end
% %export_fig 'online_simulation_accuracy' '-png' '-transparent'


