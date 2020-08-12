% Response to Reviewer Comments - EMBS 2014 Paper - Not used
load('F:\Nikunj_Data\Mahi_Experiment_Data\MS_Session2\MS_ses2_cond3_block80_average.mat')
move_epochs = Average.move_epochs;
move_erp_time = Average.move_erp_time;
RP_chans = Average.RP_chans;
time_index = find(move_erp_time == 1.00);



%% Compare first & last trials
Channel_names = {'Cz','FCz','FC2','C2'};
figure; 
for ch_ind = 1:4
    for trial_ind = 1:1
        subplot(2,2,ch_ind); hold on; grid on;
        plot(move_erp_time,move_epochs(trial_ind,:,RP_chans(ch_ind)),'b','LineWidth',1.5);
        plot(move_erp_time,move_epochs(end - trial_ind+1,:,RP_chans(ch_ind)),'Color',[0.5 0.5 0.5],'LineWidth',1.5);
    end
    set(gca,'YDir','reverse');
    title(Channel_names{ch_ind},'FontSize',12);
    
    axis([-2.5 1 -10 10]);
end
ylabel('Voltage (\muV)','FontSize',12);
xlabel('Time (sec.)','FontSize',12);
legend('First Trial','Last Trial','Location','South','Orientation','vertical');    
mtit('Stroke Patient, Passive Mode','FontSize',10);
%export_fig 'trial_comparison' '-png' '-transparent';

%% Plot Position, Velocity, Torque/ EMG epochs
        baseline_index1 = find(Torque_Epoch(1,:) == -2.5);
        baseline_index2 = find(Torque_Epoch(1,:) == -2.0);
        
        figure; subplot(3,1,2); 
        hold on; grid on;
        for k = 1:(size(Velocity_Epoch,1)-1)
            plot(Velocity_Epoch(1,1:(end-1)),Velocity_Epoch(k+1,1:(end-1)));
%                - mean(Velocity_Epoch(k+1,baseline_index1:baseline_index2)),'b');
        end
        %xlabel('Time (sec.)')
        xlim([-2.5 3]);
        ylabel('Velocity (deg/s)','FontSize',12)
        line([0 0],[-10 10],'Color','k');
        
        subplot(3,1,1); 
        hold on; grid on;
        for k = 1:(size(Position_Epoch,1)-1)
            plot(Position_Epoch(1,1:(end-1)),Position_Epoch(k+1,1:(end-1)));
%                - mean(Position_Epoch(k+1,baseline_index1:baseline_index2)),'b');
        end
        %xlabel('Time (sec.)')
        xlim([-2.5 3]);
        %ylim([-25 25])
        ylabel('Elbow Position (deg)','FontSize',12)
        v = axis;
        line([0 0],[v(3) v(4)],'Color','k');
        
        
        subplot(3,1,3);
        hold on; grid on;
        for k = 1:(size(Velocity_Epoch,1)-1)
            % No baseline correction
            plot(Torque_Epoch(1,:),Torque_Epoch(k+1,:));%- mean(Torque_Epoch(k+1,baseline_index1:baseline_index2)),'b');
        end
        xlabel('Time (sec.)','FontSize',12)
        ylabel('Torque (Nm)','FontSize',12)
        ylim([-0.3 0.4]);
        xlim([-2.5 3]);
        line([0 0],[-1 1],'Color','k');
        mtit('Subject S1, Passive Mode', 'FontSize',12);
        
%%
emg_baseline_index1 = find(emg_epochs.emg_erp_time == -2.5);
emg_baseline_index2 = find(emg_epochs.emg_erp_time == -2.0);
figure;

for k = 1:size(emg_epochs.emg_move_epochs,1)
    subplot(2,1,1); hold on;grid on;
    plot(emg_epochs.emg_erp_time,emg_epochs.emg_move_epochs(k,:,1)...
        - mean(emg_move_epochs(k,emg_baseline_index1:emg_baseline_index2,1)),'b');
    subplot(2,1,2); hold on;grid on;
    plot(emg_epochs.emg_erp_time,emg_epochs.emg_move_epochs(k,:,2)...
        - mean(emg_move_epochs(k,emg_baseline_index1:emg_baseline_index2,2)),'r');
end
