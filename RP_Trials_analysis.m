% Program for analysis of EEG data
% Program by: Nikunj Bhagat
    
raster_Fs = Average.Fs_eeg;
%raster_time = (-2:1/raster_Fs:3);
raster_time = Average.move_erp_time;
Channels_sel = [14, 32, 10, 49];%classchannels; %[14,49,53];

%% Scalp maps
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
    ScalpData = move_avg_channels(:,find(raster_time == time_stamps(tpt),1,'first')); %#ok<FNDSB>
    axes(Scalp_plot(tpt));
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

%% Raster plots for channels for all trials
%     raster_data = [move_avg_channels(53,1:31);   move_epochs(:,1:31,53)];
%     raster_time = (-2:1/raster_Fs:1);
%     raster_zscore = zscore(raster_data,0,2);
%     % Plot the rasters; Adjust parameters for plot
%     [raster_row,raster_col] = size(raster_zscore);
%     add_offset = 4;
%     raster_ylim1 = -10;
%     raster_ylim2 = (raster_row+2)*add_offset;
%     figure;
%     subplot(1,3,1);
%     %plot(raster_time,mean(raster_data),'k','LineWidth',2);
%     hold on;
%     for raster_index = 1:raster_row;
%         raster_zscore(raster_index,:) = raster_zscore(raster_index,:) + add_offset*raster_index;  % Add offset to each channel of raster plot
%         if raster_index == 1
%         plot(raster_time,raster_zscore(raster_index,:)-2,'k','LineWidth',2);
%         else 
%         plot(raster_time,raster_zscore(raster_index,:),'Color',[0.4 0.4 0.4]);
%         end
%         hold on;
%     end
%     
%     set(gca,'YDir','reverse');
%     axis([raster_time(1) raster_time(end) raster_ylim1 raster_ylim2]);
%     set(gca,'YTick',[1 6 9 12 14 16 168]);
%     set(gca,'YTickLabel',{'Average RP','Trial 1','2','.','.','.',num2str(raster_row - 1)},'FontSize',9);
%     hold on;
%     xlabel('\bfTime (sec.)','FontSize',10);
%     text(-2.5,100,'\bfSingle trials','FontSize',10,'Rotation',90); 
% %     text(-3.2,2,{'  RP';'Average'},'FontSize',10); 
%     hold off;

%% Plot voltage vs time for each snippet cluster
figure();
T_plot = tight_subplot(length(Channels_sel),3,0.05,[0.1 0.05],[0.3 0.3]);

plot_num = 1;
for ch = 1:length(Channels_sel)    
    axes(T_plot(plot_num));
    %figure;
    hold on;
    bar(1,mean(move_mean_baseline(Channels_sel(ch),:)),0.2);
    errorbar(1,mean(move_mean_baseline(Channels_sel(ch),:)),std(move_mean_baseline(Channels_sel(ch),:))/10,'LineWidth',2);
    v = axis;
    axis([0.8 1.2 v(3) v(4)]);
    set(T_plot(plot_num),'YTick',[v(3) v(4)]);
    set(T_plot(plot_num),'YTickLabel',{num2str(v(3)); num2str(v(4))});
    
    plot_num = plot_num + 1;
    axes(T_plot(plot_num));
    hold on;
    cluster = move_epochs(:,:,Channels_sel(ch));
    %cluster = 1./(1+zscore(cluster,0,2));
    %cluster = -1.*abs(cluster);
    %cluster(cluster > 0) = 0;
    cluster2 = rest_epochs(:,:,Channels_sel(ch));
    %cluster2 = 1./(1+zscore(cluster2,0,2));
    %cluster2 = -1.*abs(cluster2);
    %cluster2(cluster2 > 0) = 0;
    
    for mem = 1:size(cluster,1)
        if mem == 1
            h2 = plot(raster_time,cluster(mem,:),'Color',[0.6 0.6 0.6]);
                        
        else
            plot(raster_time,cluster(mem,:),'Color',[0.6 0.6 0.6]);
        end
        %plot(raster_time,cluster2(mem,:),'Color',[0.6 0 0.6]);
    end
    h1 = plot(raster_time,mean(cluster),'b','LineWidth',2);
    %plot(raster_time,mean(cluster2),'r','LineWidth',2);
    set(T_plot(plot_num),'XTickLabel',[]); 

    ra1 = [-5 5];
    ra2 = [-10 10];
    axis([EEG.xmin EEG.xmax ra2(1) ra2(2)]);
    
    if max(abs(move_avg_channels(Channels_sel(ch),:)+(move_SE_channels(Channels_sel(ch),:)))) <= 10 || ...
                 max(abs(move_avg_channels(Channels_sel(ch),:)-(move_SE_channels(Channels_sel(ch),:)))) <= 10
            
            axis([raster_time(1) raster_time(end) ra1(1) ra1(2)]);
            %axis([move_erp_time(1) 1 -6 6]);
            set(T_plot(plot_num),'YTick',[ra1(1) ra1(2)]);
            set(T_plot(plot_num),'YTickLabel',{num2str(ra1(1));num2str(ra1(2))});
            h3 = line([0 0],[-20 20],'Color','k','LineWidth',1.5); 
            set(T_plot(plot_num),'YDir','reverse');
            %text(2.5,-8,EEG.chanlocs(Channels_sel(ch)).labels,'FontSize',10,'FontWeight','bold');
            title([EEG.chanlocs(Channels_sel(ch)).labels],'FontSize',10);
            %set(gca,'FontWeight','bold'); 
            hold off;
            
            plot_num = plot_num + 1;
            axes(T_plot(plot_num));
            pop_erpimage(EEG,1, Channels_sel(ch),[[]],'',5,1,{},[],'','noxlabel','cbar','on','caxis',[ra1(1) ra1(2)],'cbar_title','\muV'); %EEG.chanlocs(Channels_sel(ch)).labels
            set(gca,'XTickLabel',[]); 

%             % Rest ERP single trials
%             hold on;
%             for mem = 1:size(cluster2,1)
%                 plot(raster_time,cluster2(mem,:),'Color',[0.6 0 0.6]);
%             end
%                 plot(raster_time,mean(cluster2),'r','LineWidth',2);
%                 
%             axis([raster_time(1) raster_time(end) ra1(1) ra1(2)]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(T_plot(plot_num),'YTick',[ra1(1) ra1(2)]);
%             set(T_plot(plot_num),'YTickLabel',{num2str(ra1(1));num2str(ra1(2))});
%             h3 = line([0 0],[-20 20],'Color','k','LineWidth',1.5); 
%             set(T_plot(plot_num),'YDir','reverse');
%             %text(2.5,-8,EEG.chanlocs(Channels_sel(ch)).labels,'FontSize',10,'FontWeight','bold');
%             title([EEG.chanlocs(Channels_sel(ch)).labels],'FontSize',10);
%             %set(gca,'FontWeight','bold'); 
%             hold off;

     else
            axis([raster_time(1) raster_time(end) ra2(1) ra2(2)]);
            %axis([move_erp_time(1) 1 -15 15]);
            set(T_plot(plot_num),'YTick',[ra2(1) ra2(2)]);
                       
            set(T_plot(plot_num),'YTickLabel',{num2str(ra2(1));num2str(ra2(2))});
            h3 = line([0 0],[-20 20],'Color','k','LineWidth',1.5); 
            set(T_plot(plot_num),'YDir','reverse');
            %text(2.5,-18,EEG.chanlocs(Channels_sel(ch)).labels,'FontSize',10,'FontWeight','bold');
            title([EEG.chanlocs(Channels_sel(ch)).labels],'FontSize',10);
            
            plot_num = plot_num + 1;
            axes(T_plot(plot_num)); 
            pop_erpimage(EEG,1, [Channels_sel(ch)],[[]],'',5,1,{},[],'','cbar','on','caxis',[-10 10],'cbar_title','\muV','noxlabel'); %EEG.chanlocs(Channels_sel(ch)).labels            

%             % Rest ERP single trials
%             hold on;
%             for mem = 1:size(cluster2,1)
%                 plot(raster_time,cluster2(mem,:),'Color',[0.6 0 0.6]);
%             end
%                 plot(raster_time,mean(cluster2),'r','LineWidth',2);
%                 
%             axis([raster_time(1) raster_time(end) ra2(1) ra2(2)]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(T_plot(plot_num),'YTick',[ra2(1) ra2(2)]);
%             set(T_plot(plot_num),'YTickLabel',{num2str(ra2(1));num2str(ra2(2))});
%             h3 = line([0 0],[-20 20],'Color','k','LineWidth',1.5); 
%             set(T_plot(plot_num),'YDir','reverse');
%             %text(2.5,-8,EEG.chanlocs(Channels_sel(ch)).labels,'FontSize',10,'FontWeight','bold');
%             title([EEG.chanlocs(Channels_sel(ch)).labels],'FontSize',10);
%             %set(gca,'FontWeight','bold'); 
%             hold off;
    end
        
   
%     if Channels_sel(ch) == 53
%         axis([-2 3 -15 15]);
%         
%     end

    if plot_num == 3*length(Channels_sel)
        axes(T_plot(plot_num-1)); hold on;
        xlabel('Time (sec.)','FontSize',12);
        ylabel('EEG signals (\muV)','FontSize',12);
        set(gca,'XTick',[-2 -1 0 1 2 3]);
        set(gca,'XTickLabel',{'-2'; '-1'; '0';'1';'2';'3'});
        hold off;

%         axes(T_plot(plot_num)); hold on;
%         %xlabel('Time (sec.)','FontSize',12);
%         %ylabel('EEG signals (\muV)','FontSize',12);
%         set(gca,'XTick',[-2 -1 -0.5 0 1 2 3]);
%         set(gca,'XTickLabel',{'-2'; '-1'; '-0.5'; '0';'1';'2';'3'});
    end        
    plot_num = plot_num + 1;
end
 
%mtit(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'fontsize',14,'color',[0 0 0],'xoff',-.04,'yoff',.025);

% title(sprintf('Baseline Correction Interval: %6.1f to %6.1f sec',baseline_int(1),baseline_int(2)),'FontWeight','bold');
% title('\bfAverage v/s Single RP trials','FontSize',12);
%     title(sprintf('%d Trials under Cluster %d',size(cluster,1),plot_ind2),'FontSize',12);
%    legend([h1 h2 h3],'Average RP',['Single trials (N = ' num2str(size(cluster,1)) ')'],'Movement Onset','Location','NorthEastOutside');
    %export_fig LG9_single_trials_left_hand '-png' '-transparent';
   
%% PSD analysis for trials

% Calculate ERPs for Movement Epochs
%EEG = pop_loadset( [Subject_name '_move_epochs.set'], folder_path); % read in the dataset
Fs_eeg = EEG.srate;
move_erp_time = EEG.xmin:1/Fs_eeg:EEG.xmax;
move_erp_time = (round(move_erp_time.*100))/100;

[no_channels,no_datapts,no_epochs] = size(EEG.data);
move_psd_window = [0 1.00]; %sec
NFFT =1024; % Freq_Res = Fs/NFFT
%PSD_f = zeros(1,((NFFT/2)+1));
PSD_f = Fs_eeg*(0:(NFFT/2)-1)/NFFT;
%move_epochs = zeros([no_epochs,no_datapts,no_channels]); 
%move_psd_epochs = zeros([no_epochs,((NFFT/2)),no_channels]); 

% Rectangular Window
N = 50;
w = hamming(N);

for epoch_cnt = 1:no_epochs
    for channel_cnt = 1:no_channels
        %move_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:); %EEG.data(channel_cnt,:,epoch_cnt);
        eeg_sig = move_epochs(epoch_cnt,find(move_erp_time == move_psd_window(1)):find(move_erp_time == move_psd_window(2)),channel_cnt);
        Y = fft(eeg_sig,NFFT);
        Pyy = (Y.*conj(Y))/NFFT;
        filteredsig = conv(Pyy,w);
        filteredsig = filteredsig(ceil(N/2):end-floor(N/2));
        move_psd_epochs(epoch_cnt,:,channel_cnt) = filteredsig(1:NFFT/2);
        %[move_psd_epochs(epoch_cnt,:,channel_cnt),PSD_f] = pwelch(detrend(eeg_sig),[],[],NFFT,Fs_eeg);
    end
end

for channel_cnt = 1:no_channels
    %move_psd_avg_channels(channel_cnt,:) =  mean(move_psd_epochs(:,:,channel_cnt));
    move_psd_avg_channels(channel_cnt,:) =  move_psd_epochs(1,:,channel_cnt);
    
end
%%
prepmove_psd_window = [-1.00 0]; %sec
for epoch_cnt = 1:no_epochs
    for channel_cnt = 1:no_channels
        %move_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt);
        eeg_sig = move_epochs(epoch_cnt,find(move_erp_time == prepmove_psd_window(1)):find(move_erp_time == prepmove_psd_window(2)),channel_cnt);
        Y = fft(eeg_sig,NFFT);
        Pyy = (Y.*conj(Y))/NFFT;
        filteredsig = conv(Pyy,w);
        filteredsig = filteredsig(ceil(N/2):end-floor(N/2));
        prepmove_psd_epochs(epoch_cnt,:,channel_cnt) = filteredsig(1:NFFT/2);
        %[move_psd_epochs(epoch_cnt,:,channel_cnt),PSD_f] = pwelch(detrend(eeg_sig),[],[],NFFT,Fs_eeg);
    end
end

for channel_cnt = 1:no_channels
    prepmove_psd_avg_channels(channel_cnt,:) =  mean(prepmove_psd_epochs(:,:,channel_cnt));
end

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

%% Calculate ERPs for Rest Epochs
%EEG = pop_loadset( [Subject_name '_rest_epochs.set'], folder_path); % read in the dataset
Fs_eeg = EEG.srate;
rest_erp_time = EEG.xmin:1/Fs_eeg:EEG.xmax;
rest_erp_time = (ceil(rest_erp_time.*100))/100;

[no_channels,no_datapts,no_epochs] = size(EEG.data);
rest_psd_window = [-1.0 0.0]; %sec
NFFT = 1024; % Freq_Res = Fs/NFFT
%PSD_f = zeros(1,((NFFT/2)+1));
PSD_f = Fs_eeg*(0:(NFFT/2)-1)/NFFT;
% rest_epochs = zeros([no_epochs,no_datapts,no_channels]); 
% rest_psd_epochs = zeros([no_epochs,((NFFT/2)+1),no_channels]); 


for epoch_cnt = 1:no_epochs
    for channel_cnt = 1:no_channels
        %rest_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:,epoch_cnt); 
        %rest_epochs(epoch_cnt,:,channel_cnt) = EEG.data(channel_cnt,:); 
        eeg_sig = [rest_epochs(epoch_cnt,find(rest_erp_time == rest_psd_window(1)):find(rest_erp_time == rest_psd_window(2)),channel_cnt)]; 
        Y = fft(eeg_sig,NFFT);
        Pyy = (Y.*conj(Y))/NFFT;
        filteredsig = conv(Pyy,w);
        filteredsig = filteredsig(ceil(N/2):end-floor(N/2));
        rest_psd_epochs(epoch_cnt,:,channel_cnt) = filteredsig(1:NFFT/2);
        %[rest_psd_epochs(epoch_cnt,:,channel_cnt),PSD_f] = pwelch(detrend(eeg_sig),[],[],NFFT,Fs_eeg);
    end
end

for channel_cnt = 1:no_channels
    %rest_psd_avg_channels(channel_cnt,:) =  mean(rest_psd_epochs(:,:,channel_cnt));
    rest_psd_avg_channels(channel_cnt,:) =  rest_psd_epochs(1,:,channel_cnt);
end

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

%
    %psd_Channels_nos = [13, 48, 14, 49, 15, 52, 19, 53, 20, 54];
    psd_Channels_nos = [5 14 25];
    %psd_Channels_nos = Channels_nos;
    figure; 
    %figure('units','normalized','outerposition',[0 0 1 1])
    hold on;
    plot_ind4 = 1;
    for RPind = 1:length(psd_Channels_nos)
%         if plot_ind4 == 5                 % Commented by Nikunj on Oct 31,2013
%             plot_ind4 = plot_ind4 + 1;
%         end
        subplot(1,3,plot_ind4);hold on;
        plot(PSD_f,10*log10(move_psd_avg_channels(psd_Channels_nos(RPind),:)),'b','LineWidth',2);
        plot(PSD_f,10*log10(rest_psd_avg_channels(psd_Channels_nos(RPind),:)),'r','LineWidth',2);
        %plot(PSD_f,10*log10(prepmove_psd_avg_channels(psd_Channels_nos(RPind),:)),'k','LineWidth',2);
        text(1,5,EEG.chanlocs(psd_Channels_nos(RPind)).labels,'Color','k','FontWeight','bold');
%         if max((move_avg_channels(Channels_nos(RPind),:)+(move_SE_channels(Channels_nos(RPind),:)))) >= 6 || ...
%                  min((move_avg_channels(Channels_nos(RPind),:)-(move_SE_channels(Channels_nos(RPind),:)))) <= -6
%             axis([move_erp_time(1) move_erp_time(end) -15 15]);
%             %axis([move_erp_time(1) 1 -6 6]);
%             set(gca,'FontWeight','bold');            
%         else
%             axis([move_erp_time(1) move_erp_time(end) -6 6]);
%             %axis([move_erp_time(1) 1 -15 15]);
%             
%         end
        %axis([0.01 3 0 50]);
        plot_ind4 = plot_ind4 + 1;
        grid on;
    %     xlabel('Time (sec.)')
    %     ylabel('Voltage (\muV)');
    
    end

%     subplot(2,5,8); hold on;
%     axis([0 30 0 50]);
    % topoplot([],EEG.chanlocs,'style','blank','electrodes','labels','chaninfo',EEG.chaninfo);
    subplot(1,3,2);
    hold on;
    xlabel('\bfFreq (Hz)', 'FontSize', 12);
    ylabel('\bf PSD (dB)','FontSize', 12);
    %title(['Move Window = ' num2str(move_psd_window) ' Rest Window = ' num2str(rest_psd_window)]);
    legend('PSD during Movement','PSD during Rest','PSD during Movement Preparation', 'Orientation','Horizontal');
    %export_fig LG9_PSD_vlf '-png' '-transparent';
    
%% Spectrogram
spec_chn = Channels_sel;%classchannels; %[48,14,19,53]; 
% window_len = 100;
% overlap_len = round(0.5*window_len);
% NFFT = 2^nextpow2(window_len); % Next power of 2 from length of y
% Fs = 100;
% 
% figure; [S,F,T,P] = spectrogram(move_epochs(3,:,spec_chn),window_len,overlap_len,NFFT,Fs,'yaxis');
% T = T - 2.5;
% surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
% view(0,90);
% xlabel('Time (Seconds)'); ylabel('Hz');
% colorbar
%ALLEEG(6).data(spec_chn,:,:)

figure;
T_plot = tight_subplot(2,length(spec_chn),0.01,[0.1 0.05],[0.05 0.05]);
     
for i = 1:length(spec_chn)
    axes(T_plot(i));
    %subplot(2,length(spec_chn),i);
    [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = ...
    newtimef(EEG.data(spec_chn(i),:,:),EEG.pnts,[EEG.xmin EEG.xmax]*1000,EEG.srate,[3 0.5],'freqscale','linear','maxfreq',100,'baseline',[-2500 -1500],'basenorm','off','trialbase','off',...
    'topovec',spec_chn(i),'elocs',EEG.chanlocs,'chaninfo',EEG.chaninfo,'plotitc','off','plotphasesign','off','erspmax',4,'newfig','off', ...
    'plottype','image','plotmean','off','vert',[0],'title',EEG.chanlocs(spec_chn(i)).labels);
end
for i = 1:length(spec_chn)
    %subplot(2,length(spec_chn),length(spec_chn)+i);
    axes(T_plot(length(spec_chn)+i));
    [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = ...
    newtimef(ALLEEG(CURRENTSET - 1).data(spec_chn(i),:,:),EEG.pnts,[EEG.xmin EEG.xmax]*1000,EEG.srate,[3 0.5],'freqscale','linear','maxfreq',100,'baseline',[-2500 -1500],'basenorm','off','trialbase','off',...
    'topovec',spec_chn(i),'elocs',EEG.chanlocs,'chaninfo',EEG.chaninfo,'plotitc','off','plotphasesign','off','erspmax',4,'newfig','off', ...
    'plottype','image','plotmean','off','vert',[0],'title',EEG.chanlocs(spec_chn(i)).labels);
end

%export_fig LG9_time_freq_left_side '-png' '-transparent'

%% Close loop movement detection times vs trials
% Subject_name = 'LG';
% Sess_num = 5;
% sess_trys = [2,7,8];
% folder_path = ['F:\Nikunj_Data\InMotion_Experiment_Data\LG_Session' num2str(Sess_num) '\'];  
% move_times = [];
% target_times = [];
% 
% for tr = 1:length(sess_trys)
%     move_times = [move_times;importdata([folder_path Subject_name '_s' num2str(Sess_num) '_t' num2str(sess_trys(tr)) '_Movement_events.txt'])];
%     target_times = [target_times; importdata([folder_path Subject_name '_s' num2str(Sess_num) '_t' num2str(sess_trys(tr)) '_Target_cue_events.txt'])];
% end
% 
% figure; bar((move_times(:,2) - target_times(:,2)),'0.5','hist');

%% Prob 3: Clustering with K-means algorithm

    erp_Fs = 10;
    erp_time = (-2.5:1/erp_Fs:0);

for K = 2:3
    
    num_of_clusters = K;
    num_of_itrs = 15;
    feature_space = clusterdata;
    %feature_space = feature_space(r_nk == 2,:);
    r_nk = ones(size(feature_space,1),1);
    dist = zeros(size(feature_space,1),num_of_clusters);
    J = zeros(2,num_of_itrs);

    % Step 1: Initialize Mu_k with randomly selected points
    rnd = randperm(size(feature_space,1),num_of_clusters);
    Mu_k = feature_space(rnd,:);
    %Mu_k = [move_avg_channels(14,:);rest_avg_channels(14,:);feature_space(rnd(1:num_of_clusters-2),:)];
        
    % Iterate to get minimum J
    for itr = 1:num_of_itrs
        % Step 2: E - step. Decide Mu_k & Calculate objective function J
        for datapt = 1:size(feature_space,1)
            for k = 1:num_of_clusters
                dist(datapt,k) = norm((feature_space(datapt,:) - Mu_k(k,:)),2);
            end
            J(1,itr) = J(1,itr) + dist(datapt,r_nk(datapt))^2;
        end


        % Step 3: M - step. Assign each point to a cluster with minimum distance from Mu_k
        for datapt = 1:size(feature_space,1)
            r_nk(datapt) = find(dist(datapt,:) ==  min(dist(datapt,:)));
            J(2,itr) = J(2,itr) + dist(datapt,r_nk(datapt))^2;
        end

        % Step 4: Optimize Mu_k for assigned r_nk. Set Mu_k equal to the mean of all 
        % points assigned to each cluster
        for k_flag = 1:num_of_clusters
            Mu_k(k_flag,:) = mean(feature_space(r_nk == k_flag,:));
        end
    end

    % Plot voltage vs time for each snippet cluster
    figure;
    for plot_ind2 = 1:num_of_clusters
        subplot(1,num_of_clusters,plot_ind2);
        hold on;
        cluster = feature_space((r_nk == plot_ind2),:);
        for mem = 1:size(cluster,1)
            plot(erp_time,cluster(mem,:),'Color',[0.5 0.5 0.5]);
        end
        plot(erp_time,Mu_k(plot_ind2,:),'r','LineWidth',2);
        %axis([-2 1 -10 10]);
        set(gca,'YDir','reverse');
        xlabel('Time (sec.)','FontSize',12);
        ylabel('Voltage (\muV)','FontSize',12);
        title(sprintf('%d Trials under Cluster %d',size(cluster,1),plot_ind2),'FontSize',12);
        hold off;
    end


%    Plot J vs E-M steps for convergence
    figure;
    itr_axes = 0.5:0.5:num_of_itrs;
    h3 = plot(itr_axes,J(:),'k','LineWidth',2);
    hold on;
    h1 = plot(itr_axes,J(:),'ob','LineWidth',2);
    itr_axes = 1:1:num_of_itrs;
    h2 = plot(itr_axes,J(2,itr_axes),'or','LineWidth',2);    
    legend([h1 h2],'E-step','M-step');
    set(gca,'XTick',itr_axes);
    ylabel('Objective Function J','FontSize',12');
    xlabel('Iteration Number','FontSize',12');
    title(sprintf('K-means for %d clusters',num_of_clusters),'FontSize',14);
    hold off;

end

%% EEG files for Pepe
% % 
% % 
% % % Channels = [14,53,13,15];
% % % clear Stroke;
% % % %Stroke = struct('single_trials',cell(4,1),'average_RP',cell(4,1),'time',cell(4,1),'Channel_name',cell(4,1));
% % % for i = 1:length(Channels)
% % %     Stroke(i).single_trials = move_epochs(:,:,Channels(i));
% % %     Stroke(i).average_RP = move_avg_channels(Channels(i),:);
% % %     Stroke(i).time = move_erp_time;
% % %     Stroke(i).Channel_label = EEG.chanlocs(Channels(i)).labels;
% % % end
% % 
% % Channels = [14,48,13,15];
% % clear Control;
% % %Stroke = struct('single_trials',cell(4,1),'average_RP',cell(4,1),'time',cell(4,1),'Channel_name',cell(4,1));
% % for i = 1:length(Channels)
% %     Control(i).single_trials = move_epochs(:,:,Channels(i));
% %     Control(i).average_RP = move_avg_channels(Channels(i),:);
% %     Control(i).time = move_erp_time;
% %     Control(i).Channel_label = EEG.chanlocs(Channels(i)).labels;
% % end
% % 

%%
figure;
%[S,F,T,P] = spectrogram(double(EEG.data(43,:)),[],[],512,100);
[S,F,T,P] = spectrogram(double([rest_epochs(2,:,43) move_epochs(2,:,43)]),[],0,1024,100);
surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
view(0,90);
xlabel('Time (Seconds)'); ylabel('Hz');

%% Single channels and spatial average (Mean Filtering)
move_erp_time = Average.move_erp_time;
move_epochs = Average.move_epochs;
for k = 22:22
    
    trial_num = k;
    myChannels = [14,49,32,10];
    num_ch = length(myChannels);
    single_traces = squeeze(move_epochs(trial_num,:,myChannels))';
    myAvg = mean(single_traces);

    %figure;
    for i = 1:num_ch
        subplot(3,3,i); hold on; 
        plot(move_erp_time,single_traces(i,:),'LineWidth',2);
        text(-1.5,2.5,EEG.chanlocs(myChannels(i)).labels,'Color','k','FontWeight','bold');
        set(gca,'YDir','reverse'); grid on;
        axis([-2.5 1 -10 5]);
    end
    subplot(3,3,7); hold on;
    plot(move_erp_time,myAvg,'r','LineWidth',2);
    set(gca,'YDir','reverse'); grid on;
    axis([-2.5 1 -10 5]);
    title(['Spatial Average']);
end
%export_fig spatial_average2 '-png' '-transparent';

