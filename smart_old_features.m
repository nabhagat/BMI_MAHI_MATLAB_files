

plot_eeg_emg_variability = 0;

%% Plot of EEG,EMG for movement variability

if plot_eeg_emg_variability == 1
    
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block80_velocity_epochs.mat']);
Exo_targets = Velocity_Epoch(2:end,end);

eeg_spatial_avg_up = move_ch_avg_ini(Exo_targets==3,find(move_erp_time == -2.5):find(move_erp_time == 1.0));
eeg_spatial_avg_down = move_ch_avg_ini(Exo_targets==1,find(move_erp_time == -2.5):find(move_erp_time == 1.0));
eeg_spatial_time = move_erp_time(find(move_erp_time == -2.5):find(move_erp_time == 1.0));


% For EMG - manually reject epochs, downsample to 10 Hz
biceps_up = squeeze(EEG.data(1,:,Exo_targets == 3))';
biceps_down = squeeze(EEG.data(1,:,Exo_targets == 1))';

triceps_up = squeeze(EEG.data(2,:,Exo_targets == 3))';
triceps_down = squeeze(EEG.data(2,:,Exo_targets == 1))';

% Remove baseline
eeg_spatial_avg_up = eeg_spatial_avg_up  - repmat(mean(eeg_spatial_avg_up(:,1:6),2),1,size(eeg_spatial_avg_up,2));
eeg_spatial_avg_down = eeg_spatial_avg_down  - repmat(mean(eeg_spatial_avg_down(:,1:6),2),1,size(eeg_spatial_avg_down,2));

biceps_up = biceps_up  - repmat(mean(biceps_up(:,1:6),2),1,size(biceps_up,2));
triceps_up = triceps_up  - repmat(mean(triceps_up(:,1:6),2),1,size(triceps_up,2));

% Feature Windows
eeg_old_win_up = eeg_spatial_avg_up(:,find(eeg_spatial_time == -0.7):find(eeg_spatial_time == -0.1));  %window_length = 7


%% Up Movements
f_trials = [7 21 30 4 5 16 43 13];
%figure;
for p = 1:length(f_trials) %size(eeg_spatial_avg_up,1)
    %subplot(3,3,mod(p,9)+1);
    plot(eeg_spatial_time,smooth((-1.*eeg_spatial_avg_up(f_trials(p),:)),10),'b','LineWidth',2); hold on;
    hold on; plot(eeg_spatial_time,smooth(biceps_up(f_trials(p),1:length(eeg_spatial_time)),10),'r','LineWidth',2);
    hold on; plot(eeg_spatial_time,smooth(triceps_up(f_trials(p),1:length(eeg_spatial_time)),10),'k','LineWidth',2);
    xlabel('time (sec.)','FontSize',12);
    %set(gca,'YTickLabel',[]);
    %ylabel('EEG (-ve),  EMG (+ve)','FontSize',12);
    title(['Trial No ' num2str(f_trials(p))],'FontSize',12);
    grid on;
    axis([-2.5 1 -10 15]);
    v = axis;
    legend('EEG','biceps','triceps');
    line([0 0],[v(3) v(4)],'Color',[0 0 0],'LineWidth',2);   
end


%% Down Movements
figure;
mytr = zscore(eeg_spatial_avg_down(29,:));
for p = 29:29 %size(eeg_spatial_avg_up,1)
    %subplot(3,3,mod(p,9)+1);
    %plot(eeg_spatial_time,zscore((-1.*eeg_spatial_avg_down(p,:))),'b','LineWidth',2);
    plot(eeg_spatial_time,zscore((-1.*mytr)),'b','LineWidth',2);
    hold on; plot(eeg_spatial_time,smooth(zscore(biceps_down(p,1:length(eeg_spatial_time))),10),'r','LineWidth',2);
    hold on; plot(eeg_spatial_time,smooth(zscore(triceps_down(p,1:length(eeg_spatial_time))),10),'k','LineWidth',2);
    xlabel('time (sec.)','FontSize',12);
    set(gca,'YTickLabel',[]);
    ylabel('EEG (-ve),  EMG (+ve)','FontSize',12);
    title(['Trial No ' num2str(p)],'FontSize',12);
    grid on;
    axis([-2.5 1 -3 3]);
    v = axis;
    legend('EEG','biceps','triceps');
    line([0 0],[v(3) v(4)],'Color',[0 0 0],'LineWidth',2);   
end

hold on;
plot(eeg_spatial_time(12:18),-1.*mytr(12:18),'k','LineWidth',2);
plot(eeg_spatial_time(12:18),-1.*mytr(12:18),'ok','LineWidth',2,'MarkerFaceColor','k');


%%

f_trials = [7 21 4 5 16 43 13];
figure;
for p = 1:length(f_trials) %size(eeg_spatial_avg_up,1)
    subplot(2,1,1); hold on;
%     heeg(p) = plot(eeg_spatial_time,smooth((eeg_spatial_avg_up(f_trials(p),:)),10),'b','LineWidth',2);
    heeg(p) = plot(eeg_spatial_time,eeg_spatial_avg_up(f_trials(p),:),'b','LineWidth',2);
    if p > 1
        set(heeg(p-1),'Color',[0.7 0.7 0.7]);
    end
    
    set(gca,'YDir','reverse');
    ylabel('EEG (-ve)','FontSize',12);
    %title(['Trial No ' num2str(f_trials(p))],'FontSize',12);
    grid on;
    axis([-2.5 1 -10 5]);
    v = axis;
    line([0 0],[v(3) v(4)],'Color',[0 0 0],'LineWidth',2);   
    
    subplot(2,1,2); hold on;
    hemg(p) = plot(eeg_spatial_time,smooth(biceps_up(f_trials(p),1:length(eeg_spatial_time)),10),'r','LineWidth',2);
    if p > 1
        set(hemg(p-1),'Color',[0.7 0.7 0.7]);
    end
    xlabel('time (sec.)','FontSize',12);
    %set(gca,'YTickLabel',[]);
    ylabel('EMG Envelope','FontSize',12);
    %title(['Trial No ' num2str(f_trials(p))],'FontSize',12);
    grid on;
    axis([-2.5 1 -10 15]);
    v = axis;
    line([0 0],[v(3) v(4)],'Color',[0 0 0],'LineWidth',2);   
    
end

% 
% for p = 1:length(f_trials) %size(eeg_spatial_avg_up,1)
%     subplot(2,1,1); hold on;
%     plot(-0.7:0.1:-0.1,eeg_old_win_up(f_trials(p),:),'ok','LineWidth',2);
% end
    

mt1 = 1;% find(eeg_spatial_time == find_peak_interval(1));
mt2 = find(eeg_spatial_time == find_peak_interval(2));
[min_up(:,1),min_up(:,2)] = min(eeg_spatial_avg_up(:,mt1:mt2),[],2); % value, indices
for p = 1:length(f_trials) %size(eeg_spatial_avg_up,1)
    subplot(2,1,1); hold on;
    plot(eeg_spatial_time(min_up(f_trials(p),2) - 6:min_up(f_trials(p),2)),eeg_spatial_avg_up(f_trials(p),min_up(f_trials(p),2) - 6:min_up(f_trials(p),2)),'k','LineWidth',2);
    plot(eeg_spatial_time(min_up(f_trials(p),2) - 6:min_up(f_trials(p),2)),eeg_spatial_avg_up(f_trials(p),min_up(f_trials(p),2) - 6:min_up(f_trials(p),2)),'ok','LineWidth',2,'MarkerFaceColor','k');
end

end

%% Passive mode decision times

corr_go_times = [];
wrong_go_times = [];
num_corr_go_decision = 0;
num_wrong_go_decision = 0;
num_folds =  Performance.CVO.NumTestSets;

for k = 1:num_folds
    %pred_go = find(Performance.eeg_decision{2,k} == 1);
    pred_go = Performance.eeg_decision{2,k};
    pred_go_prob_est = Performance.eeg_prob_estimates{2,k};
    
    tot_pred_go = find(pred_go == 1);
    corr_pred_go = find(pred_go(1:size(Performance.eeg_decision{2,k},1)/2) == 1);
    num_corr_go_decision = num_corr_go_decision + length(corr_pred_go);
    corr_go_times = [corr_go_times; pred_go_prob_est(corr_pred_go,1)];
    
    wrong_pred_go = tot_pred_go(length(corr_pred_go)+1:end);
    num_wrong_go_decision = num_wrong_go_decision + length(wrong_pred_go);
    wrong_go_times = [wrong_go_times; pred_go_prob_est(wrong_pred_go,1)];
end

%figure; 
histc([corr_go_times;wrong_go_times],[-2.5 -2 -1.5 -1 -0.5 0])












