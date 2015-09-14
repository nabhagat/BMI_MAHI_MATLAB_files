function [optimized_channels,rejected_channel_pool,prev_MI]  = optimized_channel_selection_backward(orig_channel_pool,move_epochs_s,rest_epochs_s,move_erp_time,rest_erp_time,find_peak_interval,reject_trial_before,...
    keep_bad_trials, window_length,rest_window)
% Optimal channel selection using mutual information criteria - backward selection
smart_move_ch_avg = [];
move_ch_avg_time = [];
smart_rest_ch_avg = [];
rest_ch_avg_time = [];
prev_MI = [];
optimized_channels = orig_channel_pool;

while ~isempty(optimized_channels)
    if ~isempty(prev_MI)
        Channel_comb_matrix = nchoosek(optimized_channels,length(optimized_channels) - 1);
    else
        % no elimination
        Channel_comb_matrix = orig_channel_pool;
    end
    %Slope_comb_matrix = zeros(total_no_combs,2*no_epochs);
    total_no_combs = size(Channel_comb_matrix,1);
    MI_comb = zeros(total_no_combs,1);
    
    for comb = 1:total_no_combs
         Smart_Features = [];
%          smart_Mu_move = []; smart_Cov_Mat = [];
%          conv_Mu_move = []; conv_Cov_Mat = [];
         bad_move_trials = []; 
         good_move_trials = [];
         
         channel_comb = Channel_comb_matrix(comb,:);
         %classchannels = classchannels(classchannels~=0);
         move_ch_avg_ini = mean(move_epochs_s(:,:,channel_comb),3);
         rest_ch_avg_ini = mean(rest_epochs_s(:,:,channel_comb),3);

         mt1 = find(move_erp_time == find_peak_interval(1));
         mt2 = find(move_erp_time == find_peak_interval(2));
         [min_avg(:,1),min_avg(:,2)] = min(move_ch_avg_ini(:,mt1:mt2),[],2); % value, indices

        for nt = 1:size(move_ch_avg_ini,1)
            if (move_erp_time(move_ch_avg_ini(nt,:) == min_avg(nt,1)) <= reject_trial_before) %|| (min_avg(nt,1) > -3)        
                %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26),'r'); hold on;
                bad_move_trials = [bad_move_trials; nt];
            else
                %plot(move_erp_time(1:26),move_ch_avg_ini(nt,1:26)); hold on;
                good_move_trials = [good_move_trials; nt];
            end
        end
       
        if keep_bad_trials == 1
            good_move_trials = 1:size(move_epochs_s,1);
            good_trials_move_ch_avg = move_ch_avg_ini(good_move_trials,:);
            good_trials_rest_ch_avg = rest_ch_avg_ini(good_move_trials,:);
        else
            % Else remove bad trials from Conventional Features, move_ch_avg and
            % rest_ch_avg
            % Commented on 6/18/14
            % Conventional_Features([bad_move_trials; (size(Conventional_Features,1)/2)+bad_move_trials],:) = [];
        %     good_trials_move_ch_avg = move_ch_avg(good_move_trials,:);
        %     good_trials_rest_ch_avg = rest_ch_avg(good_move_trials,:);
        end

        for i = 1:length(good_move_trials)
            move_window_end = find(move_ch_avg_ini(good_move_trials(i),:) == min_avg(good_move_trials(i),1)); % index of Peak(min) value
            move_window_start = move_window_end - window_length; 
            rest_window_start = find(rest_erp_time == rest_window(1));
            rest_window_end = rest_window_start + window_length;

            smart_move_ch_avg(i,:) = move_ch_avg_ini(good_move_trials(i),move_window_start:move_window_end);
            move_ch_avg_time(i,:) = move_erp_time(move_window_start:move_window_end);
            smart_rest_ch_avg(i,:) = rest_ch_avg_ini(good_move_trials(i),rest_window_start:rest_window_end); 
            rest_ch_avg_time(i,:) = rest_erp_time(rest_window_start:rest_window_end);
       end

        % Reinitialize
        no_epochs = size(smart_move_ch_avg,1);        
        data_set_labels = [ones(no_epochs,1); 2*ones(no_epochs,1)];

        %1. Slope
        Smart_Features = [(smart_move_ch_avg(:,end) - smart_move_ch_avg(:,1))./(move_ch_avg_time(:,end) - move_ch_avg_time(:,1));
                          (smart_rest_ch_avg(:,end) - smart_rest_ch_avg(:,1))./(rest_ch_avg_time(:,end) - rest_ch_avg_time(:,1))];

%         %2. Negative Peak 
%         Smart_Features = [Smart_Features [min(smart_move_ch_avg,[],2); min(smart_rest_ch_avg,[],2)]];
% 
%         %3. Area under curve
%         for ind1 = 1:size(smart_move_ch_avg,1)
%             AUC_move(ind1) = trapz(move_ch_avg_time(ind1,:),smart_move_ch_avg(ind1,:));
%             AUC_rest(ind1) = trapz(rest_ch_avg_time(ind1,:),smart_rest_ch_avg(ind1,:));
%         end
%         Smart_Features = [Smart_Features [AUC_move';AUC_rest']];
% 
%         %6. Mahalanobis distance of each trial from average over trials
%             % Use formula for computation of Mahalanobis distance
%             mahal_dist2 = zeros(2*no_epochs,1);
%             for d = 1:no_epochs
%                 mahal_dist2(d) = sqrt(mahal(smart_move_ch_avg(d,:),smart_move_ch_avg(:,:)));
%                 mahal_dist2(d + no_epochs) = sqrt(mahal(smart_rest_ch_avg(d,:),smart_move_ch_avg(:,:)));
%             end
% 
%             % Direct computation of Mahalanobis distance
%             smart_mahal_dist = zeros(2*no_epochs,1);
%             smart_Cov_Mat = cov(smart_move_ch_avg);
%             smart_Mu_move = mean(smart_move_ch_avg,1);
%             for d = 1:no_epochs
%                 x = smart_move_ch_avg(d,:);
%                 smart_mahal_dist(d) = sqrt((x-smart_Mu_move)/(smart_Cov_Mat)*(x-smart_Mu_move)');
%                 y = smart_rest_ch_avg(d,:);
%                 smart_mahal_dist(d + no_epochs) = sqrt((y-smart_Mu_move)/(smart_Cov_Mat)*(y-smart_Mu_move)');
%             end
%             Smart_Features = [Smart_Features smart_mahal_dist];
            
            disc_features = ones(size(Smart_Features));
            disc_features(Smart_Features < 0) = -1;
            MI_comb(comb) = mutualinfo(disc_features, data_set_labels);
            
    end % end for comb = 1:size(Channel_comb_matrix,1)
         
    if isempty(prev_MI)
        prev_MI = MI_comb;
        continue;
    end
        
    % Select the comb with highest mutual information
    if max(MI_comb) > prev_MI(end)
        % do something
            [max_MI,max_MI_comb] = max(MI_comb);
            optimized_channels = Channel_comb_matrix(max_MI_comb,:);
            prev_MI = [prev_MI max_MI];
    else
        % stop!!
        %classchannels = orig_channel_pool;
        rejected_channel_pool = setdiff(orig_channel_pool,optimized_channels);
        return;
    end
end
    