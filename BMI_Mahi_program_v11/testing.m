orig_Fs = 1000;
downsamp_factor = orig_Fs/10;
pkt_size = 700;
CloopClassifier = Performance;
Filtered_EEG_Downsamp = downsample(proc_eeg',downsamp_factor)';        % No baseline correction
% figure; plot(Filtered_EEG_Downsamp')
figure; plot(mean(Filtered_EEG_Downsamp,1),'k','LineWidth',2)
go_detected = [];

for sl_index = 1:length(Filtered_EEG_Downsamp)
                           sliding_window = [sliding_window(:,2:window_length) Filtered_EEG_Downsamp(:,sl_index)];
                           % Create feature vector for SVM classification
                 % 1. Compute spatial channel average (mean filtering)
                          spatial_chan_avg = mean(sliding_window,1);
                 % 2. Features - slope, -ve peak, AUC and Mahalanobis distance (in this order only) 
                          feature_vector = [(spatial_chan_avg(end) - spatial_chan_avg(1))/(window_time(end) - window_time(:,1));
                                             min(spatial_chan_avg,[],2);
                                             trapz(window_time,spatial_chan_avg);
                                             sqrt(mahal(spatial_chan_avg,CloopClassifier.move_ch_avg(:,:)))];
                         all_feature_vectors = [ all_feature_vectors feature_vector];
                 % 3. Make a prediction; Class 1: Movement; Class 2: Rest
                         [Decision_slid_win(sl_index),Acc_slid_win,Prob_slid_win(sl_index,:)] = svmpredict(DEFINE_NOGO,feature_vector',Best_BMI_classifier,'-b 1 -q');  %#ok<ASGLU>
                         
                         if (Decision_slid_win(sl_index) == DEFINE_GO) && (Prob_slid_win(sl_index,1) > prob_est_margin)        % param 1
                                valid_move_cnt = valid_move_cnt + 1;
                                go_detected = [go_detected sl_index];
        %                                     ard.digitalWrite(30,1);
        %                                     pause(0.01);
        %                                     ard.digitalWrite(30,0);
                                if valid_move_cnt >= num_valid_cnts                            % param 2
                                    GO_Probabilities = [GO_Probabilities; [Prob_slid_win(sl_index-1) Prob_slid_win(sl_index)]];
                               %****check serial port to decide if return command
        % %                                     serial_read = fread(serial_obj, 1, 'uchar');
        % % % %                                    while (serial_read ~= 'm')
        % % % %                                        serial_read = fread(serial_obj, 1, 'uchar');
        % % % %                                        display(serial_read)
        % % % %                                    end                                     
        % %                                     if  serial_read == 'm'
        % %                                              display(serial_read);

        %                                              ard.digitalWrite(32,1);
        %                                              pause(0.05);
        %                                              ard.digitalWrite(32,0);
        %                                              
        %                                             fwrite(serial_obj,'1');
                                            disp('*************Sending Move command');

        %                                             disp('Do you think that it was you that made the cursor move?');
        %                                             likert_score = [ likert_score; input('Enter Score: ')];                                                    
                                end % endif
                                move_command = [move_command; 1];
                                rest_command = [rest_command; 0];

                         else
                                move_command = [move_command; 0];
                                rest_command = [rest_command; 1];
                                valid_move_cnt = 0;           
                                disp('Rest');
                         end % endif
%                                raster_signal = [raster_signal; [Decision_slid_win, valid_move_cnt]];
                                if  valid_move_cnt >= num_valid_cnts;
                                     valid_move_cnt = 0;
                                end
                              
end % end for sl_index = 1:window_length



%% 

% HPF 
 for no_chns = 1:size(unproc_eeg,1)
    hpf_eeg(no_chns,:) = filtfilt(bhpf,ahpf,unproc_eeg(no_chns,:));
 end
    spf_eeg = L_LAP*hpf_eeg;
    
% LPF 
 for no_chns = 1:size(unproc_eeg,1)
    lpf_eeg(no_chns,:) = filtfilt(blpf,alpf,spf_eeg(no_chns,:));
 end
 
figure; 
hold on; plot(proc_eeg')
hold on; plot(mean(proc_eeg,1),'k','LineWidth',2);
hold on; plot(lpf_eeg(14,:),'--k')
hold on; plot(lpf_eeg(32,:),'--k')
hold on; plot(lpf_eeg(9,:),'--k')
hold on; plot(lpf_eeg(48,:),'--k') 

 hold on; plot(mean(lpf_eeg(Performance.classchannels,:),1),'r','LineWidth',2);
 
    
 