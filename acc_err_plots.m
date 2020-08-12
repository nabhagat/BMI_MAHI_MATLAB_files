% TNSRE Abstract Plots
mycolors_square = {'-rs','-bs','-ks','-gs','-bs'};
mycolors_circle = {'-ro','-bo','-ko','-go','-bo'};
myfacecolors = ['r','b','k','g','b'];
%load('acc_per_session.mat');
%load('err_per_session.mat');
% acc_per_session = [acc_per_session(1:2,:); acc_per_session(5,:); acc_per_session(3:4,:)]
% err_per_session = [err_per_session(1:2,:); err_per_session(5,:); err_per_session(3:4,:)]

%acc_per_session = 100.*acc_per_session;

%%
% figure();
% T_plot = tight_subplot(1,2,[0.15],[0.2 0.15],[0.15 0.15]);
% 
% for i = 1:5
%     axes(T_plot(1)); hold on;
%     if i == 2 % Subject ERWS
%         %plot([4 5],acc_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     elseif i == 3 % Subject JF
%         %plot([5],acc_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     else % Subjects LSGR, PLSH, BNBO
%         plot([3 4 5],acc_per_session(i,1:3),mycolors_square{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',8)
%     end
% end
% axes(T_plot(1)); hold on; grid on;
% axis([2.5 5.5 0 100])
% set(gca,'XTick',[3 4 5]);
% set(gca,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(gca,'YTick',[0 25 50 75 100]);
% set(gca,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',12);
% xlabel('Days','FontSize',14);
% ylabel('% TPR','FontSize',14);
% 
% 
% for i = 1:5
%     axes(T_plot(2)); hold on;
%     if i == 2 % Subject ERWS
%         %plot([4 5],err_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     elseif i == 3 % Subject JF
%         %plot([5],err_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     else % Subjects LSGR, PLSH, BNBO
%         plot([3 4 5],err_per_session(i,1:3),mycolors_circle{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',8)
%     end
% end
% axes(T_plot(2)); hold on; grid on;
% axis([2.5 5.5 0 3]);
% set(gca,'XTick',[3 4 5]);
% set(gca,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(gca,'YTick',[0 0.5 1 2 3]);
% set(gca,'YTickLabel',{'0' '0.5' '1' '2' '3'},'FontSize',12);
% xlabel('Days','FontSize',14);
% ylabel('Error/min','FontSize',14);
% leg1 = legend('BNBO','LSGR','PLSH','Orientation','Horizontal','Location','SouthOutside')
% % legtitle = get(leg1,'Title');
% % set(legtitle,'String','Subjects');

%% Multiple Y axes plots
%     acc_per_session([2 3],:) = [];
%     err_per_session([2 3],:) = [];
%     figure;
%     S_plot = tight_subplot(1,3,[0.1],[0.1 0.15],[0.1 0.1]);
% for i = 1:3
%     axes(S_plot(i)); hold on;
%     %if i == 2 % Subject ERWS
%         %plot([4 5],acc_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     %elseif i == 3 % Subject JF
%         %plot([5],acc_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     %else % Subjects LSGR, PLSH, BNBO
%         %plot([3 4 5],acc_per_session(i,1:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%         [ax,axline1,axline2] = plotyy([3 4 5],acc_per_session(i,1:3), [3 4 5], err_per_session(i,1:3),'plot','plot');
%         set(axline1,'marker','s','color',myfacecolors(i),'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i))
%         set(axline2,'marker','o','color',myfacecolors(i),'LineWidth',1.5,'LineStyle','--','MarkerFaceColor',myfacecolors(i))
%     %end
% end
% 
% title('Performance across multiple days','FontSize',12)
% xlim(ax,[2 6]);
% ylim(axline1,[0 100])
% ylim(axline2,[0 3])
% set(axline1,'XTick',[3 4 5]);
% set(axline1,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(axline1,'YTick',[0 25 50 75 100]);
% set(axline1,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',12);
% set(axline2,'YTick',[0 0.5 1 2 3]);
% set(axline2,'YTickLabel',{'0' '0.5' '1' '2' '3'},'FontSize',12);

%% Plot TPR per block
%max_block_range = 21;
max_ses3 = 4;
max_ses4 = 8;
max_ses5 = 9;
Subject_names = {'BNBO','PLSH','LSGR','ERWS','JF'};
Sess_nums = 3:5;
folder_path = ['F:\Nikunj_Data\R_analysis\'];
maxY = 50;
figure('Position',[0 -500 8*116 8*116]);
R_plot = tight_subplot(5,4,[0.025],[0.1 0.15],[0.15 0.1]);

for subj_n = 1:5
    bmi_performance = [];
    patient_performance = [];
    
    for n = 1:length(Sess_nums)
        ses_n = Sess_nums(n);
        fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
        if ~exist(fileid,'file')
            continue
        end
        cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',1,1); 
        unique_blocks = unique(cl_ses_data(:,1));
        for m = 1:length(unique_blocks)
            block_n = unique_blocks(m);
            block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
            
            ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
            ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
            block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR
            
            ind_catch_trials = find(block_performance(:,4) == 2);
            ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
            block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR
            
            time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
            Intent_per_min = 60./time_to_trigger_success_valid_trials;
            
            ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
            ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
            EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
            EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);
            
            ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,9) == 1)); % col 9 - EEG+EMG decisions
            ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,9) == 1));
            EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
            EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);
            
            bmi_performance = [bmi_performance;...
                                [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
                                 mean(Intent_per_min) std(Intent_per_min)]];
            patient_performance = [patient_performance;...
                                   ses_n block_n mean(block_performance(:,7)) std(block_performance(:,7)) ... % col 7 - Number f move attempts
                                   mean(block_performance(:,14)) std(block_performance(:,14))]; % col 14 - Likert scale 
        end % ends block_n loop
        
        %    1      2        3         4        5       6         7           8             9                   10  
        % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min)]]
        
        plotids = find(bmi_performance(:,1) == ses_n);
        switch ses_n
            case 3 % Session 3                       
                axes(R_plot(4*subj_n-3)); 
                %hold on; grid on;
                %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                axis([-1 max_ses5 0 maxY]);
                if subj_n == 5
                    set(gca,'XTickLabel',1:max_ses3,'FontSize',12);
                    %xlabel('Closed-loop Blocks','FontSize',14);
                    text(1,-20,'*p < 0.05','FontSize',12);
                else
                    set(gca,'XTickLabel','','FontSize',12);
                end
                
                if subj_n == 1
                    title('Day 3','FontSize', 12);
                end
                
                set(gca,'YTick',[0 maxY/2 maxY]);
                set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',12);
                set(gca,'XTick',[1:max_ses3]);
                h1 = ylabel(Subject_names{subj_n},'FontSize',12,'Rotation',0);
                posy = get(h1,'Position');
                set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
            case 4
                axes(R_plot(4*subj_n-2)); 
                %hold on; grid on;
                %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                axis([-1 max_ses5 0 maxY]);
                if subj_n == 5
                    set(gca,'XTickLabel',1:max_ses4,'FontSize',12);
                    xlabel('Closed-loop Blocks','FontSize',14);
                    %text(1,-33,'*p < 0.05','FontSize',12);
                else
                    set(gca,'XTickLabel','','FontSize',12);
                end
                
                if subj_n == 1
                    title('Day 4','FontSize', 12);
                end
                
                set(gca,'YTick',[0 maxY/2 maxY]);
                set(gca,'YTickLabel','');
                %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
                set(gca,'XTick',[1:max_ses4]);
            case 5
                axes(R_plot(4*subj_n-1)); 
                %hold on; grid on;
                %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                axis([-1 max_ses5+1 0 maxY]);
                if subj_n == 5
                    set(gca,'XTickLabel',1:max_ses5,'FontSize',12);
                    %xlabel('Closed-loop Blocks','FontSize',14);
                    %text(1,-33,'*p < 0.05','FontSize',12);
                else
                    set(gca,'XTickLabel','','FontSize',12);
                end
                
                if subj_n == 1
                    title('Day 5','FontSize', 12);
                end
                
                set(gca,'YTick',[0 maxY/2 maxY]);
                set(gca,'YTickLabel','');
                %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
                set(gca,'XTick',[1:max_ses5]);
            otherwise
                error('Incorrect Session Number in data.');
        end %end switch 
           
    %axis(axis);
    %hold on; arrow([1 test_regress(1)],[length(block_TPR)+1 test_regress(2)],'LineWidth',1);
    %plot([1 length(block_TPR)],test_set*mlr_TPR.Coefficients.Estimate,'-k','LineWidth',1.5)
    
    
    
    
%     set(gca,'YTick',[0 50 100]);
%     set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',12);
%     tag_text = strcat(Subject_names(i),', Intercept = ',num2str(mlr_TPR.Coefficients.Estimate(1)),', Slope =  ',num2str(mlr_TPR.Coefficients.Estimate(2)));
%     if mlr_TPR.coefTest <= 0.05
%         tag_text = strcat(tag_text,'*');
%     end
%     text(10,15,tag_text,'FontSize',12,'FontWeight','normal','Rotation',0,'BackgroundColor',[1 1 1]);
%     if i == 1
%         leg1 = legend('Day 3', 'Day 4', 'Day 5','Orientation','Horizontal','Location','NorthOutside');
%         ylabel('% True Positives','FontSize',14);
%     end
    % % legtitle = get(leg1,'Title');
    % % set(legtitle,'String','Subjects');

    end % ends ses_n loop
    axes(R_plot(4*subj_n)); 
    %hold on; grid on;
    errY = 100.*[std(bmi_performance(:,5)) std(bmi_performance(:,7));
                     std(bmi_performance(:,6)) std(bmi_performance(:,8))];
    Y = 100.*[mean(bmi_performance(:,5)) mean(bmi_performance(:,7));
                     mean(bmi_performance(:,6)) mean(bmi_performance(:,8))];
    h = barwitherr(errY,Y);
    set(h(1),'FaceColor','g');
    set(h(2),'FaceColor',1/255*[148 0 230]);
    axis([0.5 2.5 0 100]);
    set(gca,'YTick',[0 50 100]);
    set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',12);
    if subj_n == 5
       set(gca,'XTickLabel',{'TPR', 'FPR'},'FontSize',12);
       leg1 = legend(h,'EEG Only', 'EEG+EMG','Location','South','Orientation','Horizontal');
       set(leg1,'FontSize',8,'box','off');
        %set(leg1,'Box','off','PlotBoxAspectRatio',[0.6 0.5 0.5],'FontSize',10)
    else
       set(gca,'XTickLabel','','FontSize',12);
    end
    pos = get(gca,'Position');
    set(gca,'Position',[pos(1)+0.05 pos(2:4)])
    if subj_n == 5
        title('Classifier Performance','FontSize',12);
    end
    %mlr_TPR = LinearModel.fit(1:length(block_TPR),block_TPR);
    %test_regress = [ones(2,1) [1; length(block_TPR)]]*mlr_TPR.Coefficients.Estimate;
end % ends subj_n loop
% print -dtiff -r450 All_subjects_block_accuracy_bw.tif
% saveas(gca,'All_subjects_block_accuracy_bw.fig')

%% Number of attempts per block
max_block_range = 21;
Subject_names = {'BNBO','ERWS','JF','LSGR','PLSH'};
 figure('Position',[0 -500 8*116 8*116]);
Q_plot = tight_subplot(5,1,[0.025],[0.1 0.15],[0.15 0.1]);

for i = 1:1
    
    %load(['F:\Nikunj_Data\R_analysis\' Subject_names{i} '_failed_attempts_per_block.mat']);
    %dlmread()
    block_failed_attempts = failed_attempts_per_block(:,1);
    closedloop_sessions = failed_attempts_per_block(:,2);
    unique_ses = unique(closedloop_sessions);
    mlr_failed_attempts = LinearModel.fit(1:length(block_failed_attempts),block_failed_attempts);
    test_regress = [ones(2,1) [1; length(block_failed_attempts)]]*mlr_failed_attempts.Coefficients.Estimate;
    
    axes(Q_plot(i)); hold on; grid on;
    for j = 1:length(unique_ses)
        plotids = find(closedloop_sessions == unique_ses(j));
        switch unique_ses(j)
            case 3 % Session 3
                plot(plotids, block_failed_attempts(plotids),'-ro','LineWidth',1.5,'LineStyle','--','MarkerFaceColor','r','MarkerSize',8);
            case 4
                plot(plotids, block_failed_attempts(plotids),'-rs','LineWidth',1.5,'LineStyle','--','MarkerSize',8);
            case 5
                plot(plotids, block_failed_attempts(plotids),'-rs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','r','MarkerSize',8);
            otherwise
                error('Incorrect Session Number in data.');
        end
    end
    axis(axis);
    hold on; arrow([1 test_regress(1)],[length(block_failed_attempts)+1 test_regress(2)],'LineWidth',1);
    %plot([1 length(block_TPR)],test_set*mlr_TPR.Coefficients.Estimate,'-k','LineWidth',1.5)
    
    axis([1 max_block_range+1 0 8])
    set(gca,'XTick',[1:max_block_range]);
        
    if i == 5
        set(gca,'XTickLabel',1:21,'FontSize',12);
        xlabel('Closed-loop Blocks','FontSize',14);
        text(1,3,'*p < 0.05','FontSize',12);
    else
        set(gca,'XTickLabel','','FontSize',12);
    end
    set(gca,'YTick',[0 5]);
    set(gca,'YTickLabel',{'0' '5'},'FontSize',12);
    tag_text = strcat(Subject_names(i),', Intercept = ',num2str(mlr_failed_attempts.Coefficients.Estimate(1)),', Slope =  ',num2str(mlr_failed_attempts.Coefficients.Estimate(2)));
    if mlr_failed_attempts.coefTest <= 0.05
        tag_text = strcat(tag_text,'*');
    end
    text(10,5,tag_text,'FontSize',12,'FontWeight','normal','Rotation',0,'BackgroundColor',[1 1 1]);
    if i == 1
        leg1 = legend('Day 3', 'Day 4', 'Day 5','Orientation','Horizontal','Location','NorthOutside');
        ylabel('Number of Failed Movement Attempts','FontSize',14);
        set(gca,'YTick',[0 10 20]);
        set(gca,'YTickLabel',{'0' '10' '20'},'FontSize',12);
        axis([1 max_block_range+1 0 30])
    end
    % % legtitle = get(leg1,'Title');
    % % set(legtitle,'String','Subjects');
end

% print -dtiff -r450 All_subjects_block_accuracy_bw.tif
% saveas(gca,'All_subjects_block_accuracy_bw.fig')