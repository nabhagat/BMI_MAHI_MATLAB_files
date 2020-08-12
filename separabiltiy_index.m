% Program for calculating Fisher's Linear Discriminant separability index
%5. Training Classifier
move_window = [-0.7 -0.1];
rest_window = [-2.0 -1.4];
%rest_window = move_window;

classchannels = [14,13,19,53];     % Subject FJ2, JK1
%classchannels = [14,49,53,15];      % LG - left hand
%classchannels = [13,48,52,19];     % LG - right hand

Fs_eeg = EEG.srate;
move_erp_time = (round(move_erp_time.*Fs_eeg*10))./(Fs_eeg*10);
rest_erp_time = (round(rest_erp_time.*Fs_eeg*10))./(Fs_eeg*10);

% CSP Filtering
apply_csp = 0;
csp_window = [-0.7 -0.1];
%comp = 1:length(classchannels); % p = 1..2m
comp = [1 2 63 64];

if apply_csp == 1
%% 3. Common Spatial Pattern (CSP) Filter

[no_epochs,no_datapts,no_channels] = size(move_epochs);
no_channels = length(classchannels);

move_csp = zeros(no_channels,length(find(move_erp_time == csp_window(1)):find(move_erp_time == csp_window(2))),no_epochs);
rest_csp = zeros(no_channels,length(find(rest_erp_time == csp_window(1)):find(rest_erp_time == csp_window(2))),no_epochs);

for channel_cnt = 1:no_channels
    for epoch_cnt = 1:no_epochs
        move_csp(channel_cnt,:,epoch_cnt)...
            = move_epochs(epoch_cnt,find(move_erp_time == csp_window(1)):find(move_erp_time == csp_window(2)),classchannels(channel_cnt));
        rest_csp(channel_cnt,:,epoch_cnt)...
            = rest_epochs(epoch_cnt,find(rest_erp_time == csp_window(1)):find(rest_erp_time == csp_window(2)),classchannels(channel_cnt));
    end
end

% Compute normalized spatial covariance for each trial i.e. epoch
C_move = zeros(no_channels,no_channels,no_epochs);
C_rest = zeros(no_channels,no_channels,no_epochs);
for ntrial = 1:no_epochs
    Em = move_csp(:,:,ntrial);
    C_move(:,:,ntrial) = Em*Em'./trace(Em*Em');
    Er = rest_csp(:,:,ntrial);
    C_rest(:,:,ntrial) = Er*Er'./trace(Er*Er');
end

% Average spatial covariance
C_move_avg = mean(C_move,3);    
C_rest_avg = mean(C_rest,3);
% Composite spatial covariance Cc
Cc = C_move_avg + C_rest_avg;   
% Eigen value decomposition
[Uc,Lambdac] = eigs(Cc,no_channels); % Sorted in descending order of eigen values       
% Whitening transformation
disp(rank(Lambdac));
P = sqrt(pinv(Lambdac))*(Uc');  % check eig(P*Cc*P') = 1
% Apply transformation
S_move = P*C_move_avg*P';
S_rest = P*C_rest_avg*P';
[Bm,Lambda_move] = eigs(S_move,no_channels);
%[Br,Lambda_rest] = eig(S_rest);
Lambda_rest =  Bm\S_rest/(Bm');  % Check diag(Lambda_move) + diag(Lambda_rest) = 1
% Projection Matrix W
W = (Bm'*P)';
% Apply W on single trials 
Z_move = zeros(no_epochs,no_datapts,no_channels);
Z_rest = zeros(no_epochs,no_datapts,no_channels);

Zm_comp = [];
Zr_comp = [];
for i = 1:no_epochs
    Z_move(i,:,:) = (W*squeeze(move_epochs(i,:,classchannels))')';
    Z_rest(i,:,:) = (W*squeeze(rest_epochs(i,:,classchannels))')';
    Zm_comp(:,:,i) = Z_move(i,find(move_erp_time == csp_window(1)):find(move_erp_time == csp_window(2)),comp);
    Zr_comp(:,:,i) = Z_rest(i,find(rest_erp_time == csp_window(1)):find(rest_erp_time == csp_window(2)),comp);
end

fp = [squeeze(var(Zm_comp))';squeeze(var(Zr_comp))'];
Fp = [];
for j = 1:2*no_epochs
    Fp(j,:) = log(fp(j,:)./sum(fp(j,:)));
end
%move_epochs = Z_move;
%rest_epochs = Z_rest;

% figure; plot3(Fp(1:55,1),Fp(1:55,2),Fp(1:55,3),'ob');
% hold on; plot3(Fp(56:110,1),Fp(56:110,2),Fp(56:110,3),'xr');
% grid on;
else
%% 1. Extract fexture vectors from move and rest epochs
% Time Interval for training classifier
mlim1 = round(abs(EEG.xmin-(move_window(1)))*Fs_eeg+1);
mlim2 = round(abs(EEG.xmin-(move_window(2)))*Fs_eeg+1);
rlim1 = round(abs(EEG.xmin-(rest_window(1)))*Fs_eeg+1);
rlim2 = round(abs(EEG.xmin-(rest_window(2)))*Fs_eeg+1);
num_feature_per_chan = mlim2 - mlim1 + 1;

data_set = [];
peaks = zeros(2*no_epochs,length(classchannels));
% AUC = zeros(2*no_epochs,length(classchannels));

for chan_ind = 1:length(classchannels)
    move_chn = move_epochs(:,:,classchannels(chan_ind));
    rest_chn = rest_epochs(:,:,classchannels(chan_ind));
    
    feature_space = [move_chn(:,mlim1:mlim2); rest_chn(:,rlim1:rlim2)];
    peaks(:,chan_ind) = min(feature_space,[],2);
    %AUC(:,chan_ind) = [ trapz(move_erp_time(mlim1:mlim2),move_chn(:,mlim1:mlim2),2); ...
    %                    trapz(rest_erp_time(rlim1:rlim2),rest_chn(:,rlim1:rlim2),2)];
    data_set = [data_set feature_space];
end
    num_feature_per_chan = size(data_set,2)/length(classchannels);
    data_set = [data_set peaks];

data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

move_ch_avg = mean(move_epochs(:,:,classchannels),3);
rest_ch_avg = mean(rest_epochs(:,:,classchannels),3);

%1. Slope
data_set1 = [(move_ch_avg(:,mlim2) - move_ch_avg(:,mlim1))/(move_erp_time(mlim2) - move_erp_time(mlim1));
            (rest_ch_avg(:,rlim2) - rest_ch_avg(:,rlim1))/(rest_erp_time(rlim2) - rest_erp_time(rlim1))];

%2. Negative Peak 
data_set1 = [data_set1 [min(move_ch_avg(:,mlim1:mlim2),[],2); min(rest_ch_avg(:,rlim1:rlim2),[],2)]];

%3. Area under curve
data_set1 = [data_set1 [trapz(move_erp_time(mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)')';
                        trapz(rest_erp_time(rlim1:rlim2),rest_ch_avg(:,rlim1:rlim2)')']];

% %4. Mean
% data_set1 = [data_set1 [mean(move_ch_avg(:,mlim1:mlim2),2);
%                         mean(rest_ch_avg(:,rlim1:rlim2),2)]];

%5. Amplitude over interval
%data_set = [move_ch_avg(:,mlim1:mlim2); rest_ch_avg(:,rlim1:rlim2)];

%6. Mahalanobis distance of each trial from average over trials
mahal_dist = zeros(2*no_epochs,1);
for d = 1:no_epochs
    mahal_dist(d) = sqrt(mahal(move_ch_avg(d,mlim1:mlim2),move_ch_avg(:,mlim1:mlim2)));
    mahal_dist(d + no_epochs) = sqrt(mahal(rest_ch_avg(d,rlim1:rlim2),move_ch_avg(:,mlim1:mlim2)));
end
data_set1 = [data_set1 mahal_dist];
%figure; plot(data_set1(1:no_epochs,3),'ob'); hold on; plot(data_set1(no_epochs+1:2*no_epochs,3),'xr');

%%
figure; 
h = scatter3(data_set1(:,4),data_set1(:,2),data_set1(:,3),4,data_set_labels,'filled');
xlabel('Slope','FontSize',14);
ylabel('-ve Peak','FontSize',14);
zlabel('AUC','FontSize',14);
s = 0.05;
%Obtain the axes size (in axpos) in Points
currentunits = get(gca,'Units');
set(gca, 'Units', 'Points');
axpos = get(gca,'Position');
set(gca, 'Units', currentunits);

markerWidth = s/diff(xlim)*axpos(3); % Calculate Marker width in points

set(h, 'SizeData', markerWidth^4)

%title('features 1 2 3');
% figure; scatter3(data_set1(:,2),data_set1(:,3),data_set1(:,4),10,data_set_labels);
% title('features 2 3 4');
% figure; scatter(data_set1(:,1),data_set1(:,2),10,data_set_labels);


%% 2. Plot the features for move and rest classes

figure; 
valid_trials = [];
for ntrial = 1:size(data_set,1)
    subplot(1,4,1); hold on; 
    if data_set_labels(ntrial) == 1
        %plot(move_erp_time(find(move_erp_time == move_window(1)):find(move_erp_time == move_window(2))),...
         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan));
    else
        plot(move_erp_time(mlim1:mlim2),data_set(ntrial,1:num_feature_per_chan),'r')
    end
    grid on;
    set(gca,'YDir','reverse')
    axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
    hold off;
    
% %     subplot(1,4,2); hold on;
% %     if data_set_labels(ntrial) == 1
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,num_feature_per_chan+1:2*num_feature_per_chan))
% %     else
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,num_feature_per_chan+1:2*num_feature_per_chan),'r')
% %     end
% %     grid on;
% %     set(gca,'YDir','reverse')
% %     axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
% %     hold off;
% %         
% %     subplot(1,4,3); hold on;
% %     if data_set_labels(ntrial) == 1
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,2*num_feature_per_chan+1:3*num_feature_per_chan))
% %     else
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,2*num_feature_per_chan+1:3*num_feature_per_chan),'r')
% %     end
% %     grid on;
% %     set(gca,'YDir','reverse');
% %     axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
% %     %title(['Trial = ' num2str(ntrial)]);
% %     hold off;    
% %     
% %     subplot(1,4,4); hold on;
% %     if data_set_labels(ntrial) == 1
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,3*num_feature_per_chan+1:4*num_feature_per_chan))
% %     else
% %         plot(move_erp_time(mlim1:mlim2),data_set(ntrial,3*num_feature_per_chan+1:4*num_feature_per_chan),'r')
% %     end
% %     grid on;
% %     set(gca,'YDir','reverse')
% %     axis([move_erp_time(mlim1), move_erp_time(mlim2) -25 25]);
% %     %title(['Trial = ' num2str(ntrial)]);
% %     hold off;    
    
%     keep_trial = input('Keep trial? : ');
%     valid_trials = [valid_trials; keep_trial];
end
    
end
%% 4. Calculate the scatter matrices for Go and No-go conditions
% X_Go = [
%  -0.4 0.58 0.089 
%  -0.31 0.27 -0.04 
%   0.38 0.055 -0.035
%   -0.15 0.53 0.011
%  -0.35 0.47 0.034
%  0.17 0.69 0.1
%  -0.011 0.55 -0.18
%  -0.27 0.61 0.12
%  -0.065 0.49 0.0012
%  -0.12 0.054 -0.063]';


% X_Nogo = [
%   0.83 1.6 -0.014
%   1.1 1.6 0.48
%   -0.44 -0.41 0.32
%   0.047 -0.45 1.4
%   0.28 0.35 3.1
%   -0.39 -0.48 0.11
%   0.34 -0.079 0.14
%   -0.3 -0.22 2.2
%   1.1 1.2 -0.46
%  0.18 -0.11 -0.49]';

if apply_csp == 1
    data_set = Fp;
end

X_Go = data_set1(data_set_labels == 1,:)'; 
X_Nogo = data_set1(data_set_labels == 2,:)';

Mu_Go = mean(X_Go,2);
Mu_Nogo = mean(X_Nogo,2);

S_Go = (size(X_Go,2)-1)*cov(X_Go');
S_Nogo = (size(X_Nogo,2)-1)*cov(X_Nogo');

Sw = S_Go + S_Nogo; % Within group variance

Wopt = pinv(Sw)*(Mu_Go - Mu_Nogo);

y_Go = Wopt'*X_Go;
y_Nogo = Wopt'*X_Nogo;

m_Go = mean(y_Go);
m_Nogo = mean(y_Nogo);
s_Go = (length(y_Go)-1)*cov(y_Go);
s_Nogo = (length(y_Nogo)-1)*cov(y_Nogo);

f_index = (m_Go - m_Nogo)^2/(s_Go + s_Nogo);

disp(f_index);

figure;
hist(y_Nogo,ceil(sqrt(length(y_Nogo))));
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
hold on;
hist(y_Go,ceil(sqrt(length(y_Go))));
h1 = findobj(gca,'Type','patch');
set(h1,'facealpha',0.75);
legend('No-Go','Go');
%str = input('Enter label: ');
str = [Subject_name ', Sess ' num2str(Sess_num) ', #trials = ' num2str(no_epochs)];
title(['Separability Index = ' num2str(f_index) ', ' str],'FontSize',12);

%% Separability plot

% Sep_matrix = [ 0.0158 0.0174 0.0072;
%                0.0226 0.0251 0.0154;
%                0.0253 0.0303 0.0162;
%                0.0272 0.0248 0.0144;];
% 
% figure;
% hold on;
% pts = 1:4;
% h1 = plot(pts,Sep_matrix(:,1),'r',pts,Sep_matrix(:,2),'b',pts,Sep_matrix(:,3),'k');
% plot(pts,Sep_matrix(:,1),'or',pts,Sep_matrix(:,2),'xb',pts,Sep_matrix(:,3),'*k');
% set(gca,'XTick',[1,2,3,4]);
% set(gca,'XTickLabel',{'none';'CAR';'LLAP';'WAVG'});
% xlabel('Type of Spatial Filter','FontSize',12);
% ylabel('Separability Index','FontSize',12);
% legend(h1,'LG4, (55 trials)','FJ2, (100 trials)','JK1, (141,trials)');

%% Cross-Validation

CVO = cvpartition(no_epochs,'kfold',5);
cv_f_ratio = zeros(2,CVO.NumTestSets);
figure; hold on;
plot(Wopt,'ob');

for cv_index = 1:CVO.NumTestSets
    trIdx = CVO.training(cv_index);
    teIdx = CVO.test(cv_index);
    tr_X_Go = X_Go(:,trIdx);
    te_X_Go = X_Go(:,teIdx);
    tr_X_Nogo = X_Nogo(:,trIdx);
    te_X_Nogo = X_Nogo(:,teIdx);
    
    % Training
    tr_Mu_Go = mean(tr_X_Go,2);
    tr_Mu_Nogo = mean(tr_X_Nogo,2);

    tr_S_Go = (size(tr_X_Go,2)-1)*cov(tr_X_Go');
    tr_S_Nogo = (size(tr_X_Nogo,2)-1)*cov(tr_X_Nogo');

    tr_Sw = tr_S_Go + tr_S_Nogo; % Within group variance

    tr_Wopt = pinv(tr_Sw)*(tr_Mu_Go - tr_Mu_Nogo);
    plot(tr_Wopt,'x','Color',myColors(cv_index));

    tr_y_Go = tr_Wopt'*tr_X_Go;
    tr_y_Nogo = tr_Wopt'*tr_X_Nogo;

    tr_m_Go = mean(tr_y_Go);
    tr_m_Nogo = mean(tr_y_Nogo);
    tr_s_Go = (length(tr_y_Go)-1)*cov(tr_y_Go);
    tr_s_Nogo = (length(tr_y_Nogo)-1)*cov(tr_y_Nogo);

    cv_f_ratio(1,cv_index) = (tr_m_Go - tr_m_Nogo)^2/(tr_s_Go + tr_s_Nogo);
    
    % Testing 
    te_y_Go = tr_Wopt'*te_X_Go;
    te_y_Nogo = tr_Wopt'*te_X_Nogo;

    te_m_Go = mean(te_y_Go);
    te_m_Nogo = mean(te_y_Nogo);
    te_s_Go = (length(te_y_Go)-1)*cov(te_y_Go);
    te_s_Nogo = (length(te_y_Nogo)-1)*cov(te_y_Nogo);

    cv_f_ratio(2,cv_index) = (te_m_Go - te_m_Nogo)^2/(te_s_Go + te_s_Nogo);

end
hold off;
disp(cv_f_ratio);
%%
% figure; subplot(2,1,1);
% hist(tr_y_Nogo,ceil(sqrt(length(tr_y_Nogo))));
% h = findobj(gca,'Type','patch');
% set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
% hold on;
% hist(tr_y_Go,ceil(sqrt(length(tr_y_Go))));
% h1 = findobj(gca,'Type','patch');
% set(h1,'facealpha',0.75);
% legend('No-Go','Go');
% 
% subplot(2,1,2);
% hist(te_y_Nogo,ceil(sqrt(length(te_y_Nogo))));
% h = findobj(gca,'Type','patch');
% set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
% hold on;
% hist(te_y_Go,ceil(sqrt(length(te_y_Go))));
% h1 = findobj(gca,'Type','patch');
% set(h1,'facealpha',0.75);
% legend('No-Go','Go');

%% Do processing here
    %1. Logarithm
    % move_chn = log(abs(move_chn));
    % rest_chn = log(abs(rest_chn));
    
    %2. 1/1+x
    %move_chn = 1./(1 + abs(move_chn));
    %rest_chn = 1./(1 + abs(rest_chn));
    
    %3. Slope
    %move_chn = diff(move_chn,1,2).*Fs_eeg;
    %rest_chn = diff(rest_chn,1,2).*Fs_eeg;
    
    %4. Normalize
%     for nd = 1:no_epochs
%         move_chn(nd,:) = move_chn(nd,:)./min(move_chn(nd,1:find(move_erp_time == 0)),[],2);
%         rest_chn(nd,:) = rest_chn(nd,:)./min(rest_chn(nd,1:find(rest_erp_time == 0)),[],2);
%     end
















