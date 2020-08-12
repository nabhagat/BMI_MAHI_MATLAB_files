% % % Analysis of movement and rest epochs
% % % close all; 
% % % clear all;
% % 
% % %load('move_epochs.mat');
% % %load('rest_epochs.mat');
% % 
% % Cz_move = move_epochs(:,:,14);
% % Cz_rest = rest_epochs(:,:,14);
% % [move_no_trials,no_samples] = size(Cz_move);
% % plot_time = -2:1/10:(1-(1/10));
% % 
% % % % std_move = std(Cz_move);
% % % % mean_move = mean(Cz_move);
% % % % std_rest = std(Cz_rest);
% % % % mean_rest = mean(Cz_rest);
% % 
% % % % figure;hold on;
% % % % plot(plot_time,mean_move,'b','LineWidth',2);
% % % % % plot(plot_time,mean_move+std_move,'--b');
% % % % % plot(plot_time,mean_move-std_move,'--b');
% % % % jbfill(plot_time,mean_move+std_move,mean_move-std_move,'b','k',1,0.2)
% % % % hold on;
% % % % plot(plot_time,mean_rest,'r','LineWidth',2);
% % % % % plot(plot_time,mean_rest+std_rest,'--r');
% % % % % plot(plot_time,mean_rest-std_rest,'--r');
% % % % jbfill(plot_time,mean_rest+std_rest,mean_rest-std_rest,'r','k',1,0.1)
% % % % hold off;
% % 
% % %% Calculate Instantenous slope
% % % lim1 = abs(-2-(-0.7))*10;
% % % lim2 = abs(-2-(-0.2))*10;
% % % 
% % % move_slope_750_250 = zeros(move_no_trials,1);
% % % for k = 1:move_no_trials
% % %     move_slope_750_250(k) = ((Cz_move(k,lim1) - Cz_move(k,lim2))/(-0.7+0.2));             % slope = (y2-y1)/(x2-x1)
% % %     move_diff(k,:) = diff(Cz_move(k,:));
% % % end
% % % 
% % % [rest_no_trials,no_samples] = size(Cz_rest);
% % % rest_slope_750_250 = zeros(rest_no_trials,1);
% % % for k = 1:rest_no_trials
% % %     rest_slope_750_250(k) = ((Cz_rest(k,lim1) - Cz_rest(k,lim2))/(-0.7+0.2));             % slope = (y2-y1)/(x2-x1)
% % %     rest_diff(k,:) = diff(Cz_rest(k,:));
% % % end
% % % 
% % % % figure;
% % % % plot(move_slope_750_250,'ob');
% % % % hold on;
% % % % plot(rest_slope_750_250,'*r');
% % % % hold off;
% % % figure;
% % % plot_time = -2:1/10:(1-(1/10));
% % % subplot(2,1,1);
% % % plot(plot_time,Cz_move(1,:),'b',plot_time,Cz_rest(4,:),'r','LineWidth',2);
% % % subplot(2,1,2);
% % % hist2(move_slope_750_250,rest_slope_750_250);
% % 
% % %%
% % avg_sensitivity = [];
% % avg_specificity = [];
% % avg_accur       = [];
% % num_channels    = [];
% % avg_TPR = [];
% % avg_FPR = [];
% % %% LDA
% % mlim1 = abs(-2-(-0.6))*10+1;
% % mlim2 = abs(-2-(-0.2))*10+1;
% % rlim1 = round(abs(-1-(-0.4))*10);
% % rlim2 = abs(-1-(-0.0))*10;
% % num_diff = 1;
% % gain = 10^num_diff;
% % if gain == 0
% %     gain = 1;
% % end
% % classchannels = [14,19,48,9,53,20];
% % num_feature_per_chan = mlim2 - mlim1 + 1 - num_diff;
% % train_range = 100;
% % test_range = 0;
% % 
% % train_set = zeros(2*train_range,length(classchannels)*(length(mlim1:mlim2)-num_diff));
% % test_set = zeros(2*test_range,length(classchannels)*(length(mlim1:mlim2)-num_diff));
% % train_labels = [];
% % 
% % for train_ind = 1:train_range
% %     if num_diff == 0
% % %         train_set(train_ind,:) = [move_epochs(train_ind,mlim1:mlim2,classchannels(1))...
% % %                         move_epochs(train_ind,mlim1:mlim2,classchannels(2))...
% % %                         move_epochs(train_ind,mlim1:mlim2,classchannels(3))];
% %         for chan_ind = 1:length(classchannels)
% %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% %             ra2 = chan_ind*num_feature_per_chan;
% %             train_set(train_ind,ra1:ra2) = move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind));
% %         end
% %             
% %     else
% % %         train_set(train_ind,:) = [diff(move_epochs(train_ind,mlim1:mlim2,classchannels(1)),num_diff,2)...
% % %                         diff(move_epochs(train_ind,mlim1:mlim2,classchannels(2)),num_diff,2)...
% % %                         diff(move_epochs(train_ind,mlim1:mlim2,classchannels(3)),num_diff,2)].*gain;
% %         for chan_ind = 1:length(classchannels)
% %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% %             ra2 = chan_ind*num_feature_per_chan;
% %             train_set(train_ind,ra1:ra2) = diff(move_epochs(train_ind,mlim1:mlim2,classchannels(chan_ind)),num_diff,2).*gain;
% %         end
% % 
% %     end
% %     
% %     train_labels(train_ind,1) = 1;
% %     
% %     if num_diff == 0
% % %         train_set(train_ind+train_range,:) = [rest_epochs(train_ind,mlim1:mlim2,classchannels(1))...
% % %                        rest_epochs(train_ind,mlim1:mlim2,classchannels(2))...
% % %                        rest_epochs(train_ind,mlim1:mlim2,classchannels(3))];
% % %                    
% %        for chan_ind = 1:length(classchannels)
% %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% %             ra2 = chan_ind*num_feature_per_chan;
% %             train_set(train_ind+train_range,ra1:ra2) = rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind));
% %        end
% %         
% %     else 
% % %         train_set(train_ind+train_range,:) = [diff(rest_epochs(train_ind,mlim1:mlim2,classchannels(1)),num_diff,2)...
% % %                        diff(rest_epochs(train_ind,mlim1:mlim2,classchannels(2)),num_diff,2)...
% % %                        diff(rest_epochs(train_ind,mlim1:mlim2,classchannels(3)),num_diff,2)].*gain;
% % 
% %         for chan_ind = 1:length(classchannels)
% %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% %             ra2 = chan_ind*num_feature_per_chan;
% %             train_set(train_ind+train_range,ra1:ra2) = diff(rest_epochs(train_ind,rlim1:rlim2,classchannels(chan_ind)),num_diff,2).*gain;
% %         end
% %        
% %     end
% % %     train_set(train_ind+train_range,:) = zeros(1,15); 
% %     train_labels(train_ind+train_range,1) = 2;
% % end
% % 
% % % % Commented on 8 Sept, 2013
% % % for test_ind = 1:test_range
% % %     if num_diff == 0
% % % %         test_set(test_ind,:) = [move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(1))...
% % % %                         move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(2))...
% % % %                         move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(3))];
% % %         for chan_ind = 1:length(classchannels)
% % %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% % %             ra2 = chan_ind*num_feature_per_chan;
% % %             test_set(test_ind,ra1:ra2) = move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(chan_ind));
% % %         end
% % %     else
% % % %         test_set(test_ind,:) = [diff(move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(1)),num_diff,2)...
% % % %                         diff(move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(2)),num_diff,2)...
% % % %                         diff(move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(3)),num_diff,2)].*gain;
% % %         for chan_ind = 1:length(classchannels)
% % %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% % %             ra2 = chan_ind*num_feature_per_chan;
% % %             test_set(test_ind,ra1:ra2) = diff(move_epochs(train_range+test_ind,mlim1:mlim2,classchannels(chan_ind)),num_diff,2).*gain;
% % %         end
% % %     end
% % %     
% % %     if num_diff == 0
% % % %         test_set(test_ind+test_range,:) = [rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(1))...
% % % %                         rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(2))...
% % % %                         rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(3))];
% % %         for chan_ind = 1:length(classchannels)
% % %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% % %             ra2 = chan_ind*num_feature_per_chan;
% % %             test_set(test_ind+test_range,ra1:ra2) = rest_epochs(train_range+test_ind,rlim1:rlim2,classchannels(chan_ind));
% % %         end
% % %     else
% % % %         test_set(test_ind+test_range,:) = [diff(rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(1)),num_diff,2)...
% % % %                         diff(rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(2)),num_diff,2)...
% % % %                         diff(rest_epochs(train_range+test_ind,mlim1:mlim2,classchannels(3)),num_diff,2)].*gain;
% % %         for chan_ind = 1:length(classchannels)
% % %             ra1 = (chan_ind - 1)*num_feature_per_chan + 1;
% % %             ra2 = chan_ind*num_feature_per_chan;
% % %             test_set(test_ind+test_range,ra1:ra2) = diff(rest_epochs(train_range+test_ind,rlim1:rlim2,classchannels(chan_ind)),num_diff,2).*gain;
% % %         end
% % %     end
% % % %    test_set(test_ind+test_range,:) = zeros(1,15);
% % % end
% % % true_labels = [ones(test_range,1);2*ones(test_range,1)];
% % % % Commented on 8 Sept, 2013
% % 
% % 
% % 
% % CVO = cvpartition(200,'kfold',5);
% % sensitivity = zeros(1,CVO.NumTestSets);
% % specificity = zeros(1,CVO.NumTestSets);
% % accuracy = zeros(1,CVO.NumTestSets);
% % TPR = zeros(1,CVO.NumTestSets);
% % FPR = zeros(1,CVO.NumTestSets);
% % 
% % for i = 1:CVO.NumTestSets
% %     trIdx = CVO.training(i);
% %     teIdx = CVO.test(i);
% %     %ytest = classify(observes(teIdx,:),observes(trIdx,:),...
% %     %        labels(trIdx,:));
% %     [test_labels, Error, Posterior, LogP, OutputCoefficients] = ...
% %        classify(train_set(teIdx,:),train_set(trIdx,:),train_labels(trIdx,:), 'linear');
% % 
% %     %err(i) = sum(~strcmp(ytest,labels(teIdx)));
% %     CM = confusionmat(train_labels(teIdx,:),test_labels);
% %     sensitivity(i) = CM(1,1)/(CM(1,1)+CM(1,2));
% %     specificity(i) = CM(2,2)/(CM(2,2)+CM(2,1));
% %     TPR(i) = CM(1,1)/(CM(1,1)+CM(1,2));
% %     FPR(i) = CM(2,1)/(CM(2,2)+CM(2,1));
% %     accur(i) = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2))*100;
% %     
% % end
% % 
% % avg_sensitivity = [avg_sensitivity; mean(sensitivity)];
% % avg_specificity = [avg_specificity; mean(specificity)];
% % avg_accur       = [avg_accur; mean(accur)];
% % avg_TPR         = [avg_TPR; mean(TPR)];
% % avg_FPR         = [avg_FPR; mean(FPR)];
% % num_channels    = [num_channels; length(classchannels)];
% % %num_channels    = [num_channels; num_feature_per_chan];
% % %%
% % figure; hold on; grid on;
% % plot(num_channels,100*avg_sensitivity,'ob-');
% % plot(num_channels,100*avg_specificity,'or-');
% % %plot(num_channels,avg_accur,'ok-');
% % figaxes = axis;
% % axis([figaxes(1) figaxes(2) 0 100]);
% % set(gca,'XDir','reverse');
% % xlabel('Number of time lags')
% % ylabel('Percentage')
% % legend('Sensitivity','Specificity')
% % title('Classifier Performance')
% % 
% % 
% % %% Create scatter3 colormap
% % % Colormap = zeros(length(train_labels),3);
% % % for j = 1:length(train_labels)
% % %     switch train_labels(j,1)
% % %         case 1
% % %             Colormap(j,:) = [0 1 0];
% % %         case 2
% % %             Colormap(j,:) = [1 0 0];
% % %         case 'f'
% % %             Colormap(j,:) = [0 0 1];
% % %         otherwise
% % %             Colormap(j,:) = [0 0 0];
% % %     end
% % % end
% % % 
% % % figure;
% % % scatter3(train_set(:,1), train_set(:,2), train_set(:,3), 2, Colormap);
% % % hold on;
% % % Xlimits = get(gca,'XLim');  %dont worry about this...
% % % Ylimits = get(gca,'YLim');  %or this... just getting the axis limits.
% % % Zlimits = get(gca,'ZLim');  
% % 
% % %%
% % 
% % %  [test_labels, Error, Posterior, LogP, OutputCoefficients] = ...
% % %     classify(test_set,train_set,train_labels, 'linear');
% % 
% % % K12 = OutputCoefficients(1,2).const;
% % % L12 = OutputCoefficients(1,2).linear;
% % % f12 = @(x,y,z) K12 + [x y z]*L12;
% % % f12plane = @(x,y) ((K12+L12(1)*x+L12(2)*y)/(-L12(3)));
% % % 
% % % X = Xlimits(1):1:20;
% % % [X,Y] = meshgrid(X);
% % % Z = f12plane(X,Y);
% % % C = ones(size(Z));
% % % surf(X,Y,Z,C);
% % 
% % 
% % 
% % %% Percentage Changes in slope
% % 
% % load('move_epochs.mat');
% % load('rest_epochs.mat');
% % 
% % mlim1 = abs(-2-(-0.6))*10;
% % mlim2 = abs(-2-(-0.2))*10;
% % % Cz_move, train_set defined earlier
% % % Cz_rest defined earlier
% % Cz_move_win = move_epochs(:,mlim1:mlim2,14);
% % Cz_rest_win = rest_epochs(:,mlim1:mlim2,14);
% % 
% % Cz_slope_move = diff(Cz_move_win,1,2).*10;
% % Cz_slope_rest = diff(Cz_rest_win,1,2).*10;
% % valid_move = [];
% % 
% % % find valid slopes
% % valid_cnt = 0;
% % 
% % for slope_ind = 1:length(Cz_slope_move)
% %     if Cz_slope_move(slope_ind,1) > Cz_slope_move(slope_ind,2)
% %         if Cz_slope_move(slope_ind,2) > Cz_slope_move(slope_ind,3)
% %             if Cz_slope_move(slope_ind,3) > Cz_slope_move(slope_ind,4)
% %                 %if Cz_slope_move(slope_ind,4) > Cz_slope_move(slope_ind,5)
% %                     valid_cnt = valid_cnt + 1;
% %                     valid_move = [valid_move; Cz_slope_move(slope_ind,:)];
% %                     %invalid_rest = [invalid_rest; Cz_slope_rest(slope_ind,:)];
% %                 %end
% %             end
% %         end
% %     end
% % end
% % 
% % invalid_rest = [];
% % invalid_cnt = 0;
% % for slope_ind = 1:length(Cz_slope_rest)
% %     if Cz_slope_rest(slope_ind,1) > Cz_slope_rest(slope_ind,2)
% %         if Cz_slope_rest(slope_ind,2) > Cz_slope_rest(slope_ind,3)
% %             if Cz_slope_rest(slope_ind,3) > Cz_slope_rest(slope_ind,4)
% %                 %if Cz_slope_rest(slope_ind,4) > Cz_slope_rest(slope_ind,5)
% %                     invalid_cnt = invalid_cnt + 1;
% %                     %valid_move = [valid_move; Cz_slope_rest(slope_ind,:)];
% %                     invalid_rest = [invalid_rest; Cz_slope_rest(slope_ind,:)];
% %                 %end
% %             end
% %         end
% %     end
% % end
% % 
% % slope_percent = diff(valid_move,1,2);
% % 
% % % Percentage change in slope
% % for percent_ind = 1:3
% %     slope_percent(:,percent_ind) = (slope_percent(:,percent_ind)./valid_move(:,percent_ind)).*100;
% % end
% % 
% % 
% % %% My figure
% % mlim1 = abs(-2-(-0.7))*10+1;
% % mlim2 = abs(-2-(-0.2))*10+1;
% % plot_time = -2:1/10:(1-(1/10));
% % 
% % figure;
% % plot(plot_time,move_avg_channels(14,:),'b','LineWidth',2)
% % set(gca,'YDir','reverse');
% % % axis([-2 1 -8 2])
% % %line([0 0],[-8 4],'Color','k','LineWidth',2);
% % xlabel('Time (sec.)')
% % ylabel('Voltage (\muV)');
% % 
% % hold on;
% % plot(plot_time(mlim1:mlim2),move_avg_channels(14,mlim1:mlim2),'ok')
% % jbfill((-0.7:0.1:-0.2),4*ones(1,6),-12*ones(1,6),'r','k',1,0.2)
% % 
% % %%  MY figure 2
% % rlim1 = abs(-1-(-0.5))*10;
% % rlim2 = abs(-1-(-0.0))*10;
% % plot_time = -5:1/10:(-4-(1/10));
% % 
% % %figure;
% % hold on;
% % plot(plot_time,rest_avg_channels(14,:),'b','LineWidth',2)
% % set(gca,'YDir','reverse');
% % line([-4 -4],[-8 4],'Color','r','LineWidth',2);
% % xlabel('Time (sec.)')
% % ylabel('Voltage (\muV)');
% % line([-5 2],[0 0],'Color','k','LineWidth',2);
% % hold on;
% % plot(plot_time(rlim1:rlim2),rest_avg_channels(14,rlim1:rlim2),'ok')
% % jbfill((-4.5:0.1:-4.1),4*ones(1,6),-12*ones(1,6),'b','k',1,0.2)

%% Baseline correction
% feature_space = [abs(mean(data_set(:,15:21),2)), abs(mean(data_set(:,22:28),2)), abs(mean(data_set(:,29:35),2))];
% feature_space = [min(data_set(:,15:21),[],2), min(data_set(:,22:28),[],2) min(data_set(:,29:35),[],2)];

% Apply simple linear regression to calculate slope for each trial
% t_interval =  move_window(1):0.1:move_window(2);
% 
% [R1,M1,B1] = regression(repmat(t_interval,size(data_set,1),1),data_set(:,1:7));
% [R2,M2,B2] = regression(repmat(t_interval,size(data_set,1),1),data_set(:,8:14));
% [R3,M3,B3] = regression(repmat(t_interval,size(data_set,1),1),data_set(:,15:21));
% 
% % feature_space = [B1,B2,B3];

%%
% Create scatter3 colormap
feature_space = [X_Go';X_Nogo';Wopt';[0 0 0]];
feature_labels = [ones(10,1);2*ones(10,1);3;3];

feature_space = [feature_space; y_Go'*Wopt'; y_Nogo'*Wopt'];
feature_labels = [feature_labels; feature_labels(1:end-2)];

Colormap_result = [zeros(length(feature_labels),3)];
for j = 1:length(feature_labels)
    switch feature_labels(j,1)
        case 1
            Colormap_result(j,:) = [0 0 0];
        case 2
            Colormap_result(j,:) = [0.6 0.6 0.6];
        otherwise
            Colormap_result(j,:) = [1 0 0];
    end
end
figure; 
h = scatter3(feature_space(:,1),feature_space(:,2),feature_space(:,3),4,Colormap_result,'filled');
xlabel('chan Cz','FontSize',14);
ylabel('chan C1','FontSize',14);
zlabel('chan C2','FontSize',14);
%title('Mean values of baseline corrected rest and move trials','FontSize',14);
axis([-1 1 -1 1 -1 1]);

s = 0.05;
%Obtain the axes size (in axpos) in Points
currentunits = get(gca,'Units');
set(gca, 'Units', 'Points');
axpos = get(gca,'Position');
set(gca, 'Units', currentunits);

markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points

set(h, 'SizeData', markerWidth^2)

line([-5*Wopt(1) 5*Wopt(1)],[-5*Wopt(2) 5*Wopt(2)],[-5*Wopt(3) 5*Wopt(3)],'Color','b','LineWidth',2);

%% 
% Create scatter3 colormap
% Use k-means clustering
[IDx,Cent] = kmeans(feature_space,2);

Colormap_result = [zeros(length(IDx),3)];
for j = 1:length(IDx)
    switch IDx(j,1)
        case 1
            Colormap_result(j,:) = [0 0 1];         % blue means move, black goes to blue
        case 2  
            Colormap_result(j,:) = [1 0 0];         % red means rest, gray goes to red
        otherwise
            Colormap_result(j,:) = [0 0 0];
    end
end
% figure; 
hold on;
h = scatter3(feature_space(:,1),feature_space(:,2),feature_space(:,3),4,Colormap_result);
xlabel('chan Cz','FontSize',14);
ylabel('chan C1','FontSize',14);
zlabel('chan C2','FontSize',14);
title('mean','FontSize',14);
axis([-10 10 -10 10 -10 10]);

s = 0.5;
%Obtain the axes size (in axpos) in Points
currentunits = get(gca,'Units');
set(gca, 'Units', 'Points');
axpos = get(gca,'Position');
set(gca, 'Units', currentunits);
markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points
set(h, 'SizeData', markerWidth^2)

% Plot centroids
hold on;
plot3(Cent(1,1),Cent(1,2),Cent(1,3),'ob','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[0 0 1]);
plot3(Cent(2,1),Cent(2,2),Cent(2,3),'or','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[1 0 0]);
