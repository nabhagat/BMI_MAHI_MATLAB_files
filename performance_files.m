% Subject Details
clear;
clc;
Subject_name = 'BNBO';
Sess_num = '2';
Cond_num = 3;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;
max_day1_trials = 80; % Remains same for all subjects
folder_path = ['F:\Nikunj_Data\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
Performance = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
Performance_conv = importdata([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_conventional.mat']);      

remove_corrupted_epochs = [];
%remove_corrupted_epochs = [remove_corrupted_epochs 41 125 153]; %ERWS_ses2_cond3_block160
%remove_corrupted_epochs = [remove_corrupted_epochs 21 42 111 112 140 147]; %PLSH_ses2_cond1_block160
%remove_corrupted_epochs = [remove_corrupted_epochs 41 43 52 54 59]; %LSGR_ses1_cond3
%remove_corrupted_epochs = [remove_corrupted_epochs 83];  % LSGR_ses2b_cond3

all_good_trials = Performance.good_move_trials; 
for i = 1:length(remove_corrupted_epochs)
    rm = remove_corrupted_epochs(i);
    above_trials = find(all_good_trials >= rm);
    all_good_trials(above_trials) = all_good_trials(above_trials) + 1;  
end

true_positive_decision_latency = [];
num_true_positives = [];
num_true_negatives = [];
for k = 1:Performance.CVO.NumTestSets
    fold_prob_estimate = Performance.eeg_prob_estimates{k};
    true_positives = find(fold_prob_estimate(1:Performance.CVO.TestSize(k),1) >= 0.5);
    true_negatives = Performance.CVO.TestSize(k) + find(fold_prob_estimate(Performance.CVO.TestSize(k)+1:end,1) < 0.5);
    true_positive_decision_latency = [true_positive_decision_latency; fold_prob_estimate(true_positives,3)];
    
    num_true_positives = [num_true_positives; length(true_positives)];
    num_true_negatives = [num_true_negatives; length(true_negatives)];
end

conv_true_positive_decision_latency = [];
conv_num_true_positives = [];
conv_num_true_negatives = [];
for k = 1:Performance_conv.CVO.NumTestSets
    fold_prob_estimate = Performance_conv.All_eeg_prob_estimates{Performance_conv.conv_opt_wl_ind}{k};
    true_positives = find(fold_prob_estimate(1:Performance_conv.CVO.TestSize(k),1) >= 0.5);
    true_negatives = Performance_conv.CVO.TestSize(k) + find(fold_prob_estimate(Performance_conv.CVO.TestSize(k)+1:end,1) < 0.5);
    conv_true_positive_decision_latency = [conv_true_positive_decision_latency; fold_prob_estimate(true_positives,3)];
    
    conv_num_true_positives = [conv_num_true_positives; length(true_positives)];
    conv_num_true_negatives = [conv_num_true_negatives; length(true_negatives)];
end

[smart_latency_hist, smart_latency_X]= hist(true_positive_decision_latency);
figure; bar(smart_latency_X,smart_latency_hist,'r');
[conv_latency_hist,conv_latency_X] = hist(conv_true_positive_decision_latency);
hold on; bar(conv_latency_X,conv_latency_hist,'g');

t20 = length(find(all_good_trials <= 20)); 
t20_40 = length(find(all_good_trials > 20 & all_good_trials <= 40));
t40_60 = length(find(all_good_trials > 40 & all_good_trials <= 60));
t60_80 = length(find(all_good_trials > 60 & all_good_trials <= 80));
t80_100 = length(find(all_good_trials > 80 & all_good_trials <= 100));
t100_120 = length(find(all_good_trials > 100 & all_good_trials <= 120));
t120_140 = length(find(all_good_trials > 120 & all_good_trials <= 140));
t140_160 = length(find(all_good_trials > 140 & all_good_trials <= 160));


disp([size(Performance.Conventional_Features,1)/2, size(Performance.good_move_trials,1), length(find(all_good_trials <= 80)), ...
    Performance.smart_window_length, length(true_positive_decision_latency),...
    mean(true_positive_decision_latency), std(true_positive_decision_latency)...
    Performance_conv.conv_window_length,  length(conv_true_positive_decision_latency),...
    mean(conv_true_positive_decision_latency), std(conv_true_positive_decision_latency) ...
    ]);

disp('Block-wise: ');
disp([t20 t20_40 t40_60 t60_80 t80_100 t100_120 t120_140 t140_160])

figure('Position',[1050 1300 5*116 2.5*116]); 
%x=1:8;
y=abs([t20 t20_40 t40_60 t60_80; t80_100 t100_120 t120_140 t140_160]);
bar(y)
% for i1=1:numel(y)
%     text(x(i1),y(i1),num2str(y(i1),'%0.2f'),...
%                'HorizontalAlignment','center',...
%                'VerticalAlignment','bottom')
% end
ylim([0 20])
print('-dpng', '-r300', [Subject_name '_cond' num2str(Cond_num) '.png']); 


%Performance.eeg_accur = (num_true_positives + num_true_negatives)'/Performance.CVO.TestSize


% http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/homepage.htm
% http://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
%means = [-0.781, -0.8663, -0.6823, -0.877, -0.7339, -0.3869];   % smart latencies
%trials = [84, 89, 79, 74, 124, 130];    % smart intent detections
%means = [-0.9197, -0.9575, -0.9128, -0.8856, -1.1296, -0.8748];     % conventional window latencies
%trials = [132, 153, 149, 156, 159, 155];                            % conventional intent detections

means = [-0.4968, -0.4565, -0.6825, -0.4837, -0.3459, -0.6087, -0.3068, 0.2951];        % Closed-loop BMI latencies
trials = [69, 106, 108, 88, 76, 57, 73, 107];

indexes = 1:8;
%indexes = [1 3 5 7];
%indexes = [2 4 6 8];
means = means(indexes);
trials = trials(indexes);

N = length(trials);
weighted_means = sum(means.*trials)/sum(trials);
weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));




