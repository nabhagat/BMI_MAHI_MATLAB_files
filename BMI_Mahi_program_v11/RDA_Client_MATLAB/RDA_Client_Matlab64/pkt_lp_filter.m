function Yn = pkt_lp_filter(Xn,Xn_k,Yn_k,samp_freq)
% Low Pass Filter EEG data packets with cutoff frequency of 1 Hz. This is a 4th order
% butterworth filter designed using fdatool. 
    
    % To verify that this filter is actually working with the EEG data: 
    % 1. Run the Brain Vision Recorder software in Simulated Amplifier mode. 
    % 2. Generate a N channels as a sine wave. 
    
    % 3. Modify output of CAR as 
    %                                  tf = 1/500:1/500:0.5;
    %                                 car(1,:) = car(1,:) + sin(2*pi*100*(tf));
    
    % 4. Save the car and lp_filt_pkt variables as unprocessed_eeg and processed_eeg variables, respectively from 
    %     the RDA client program. 
    % 5. Run the Inmotion_decoding program in closeloop testing mode. 
    % 6. View the proc_eeg & unproc_eeg offline to check if the high frequency noise has been filtered 
    % 7. You can also use pmtm or pwelch.
%% ***********Revisions 
%   5/15/2014 - Added switch statement to choose raw EEG sampling frequency. This affects the filter coefficients    

%--------------------------------------------------------------------------------------------------
    
    %[b,a] = butter(4,(1/(samp_freq/2)),'low')
    switch samp_freq
        case 500
                b = [1.53324549584388e-09,6.13298198337553e-09,9.19947297506329e-09,6.13298198337553e-09,1.53324549584388e-09];      % numerator coefficients
                a = [1,-3.96716259594885,5.90202586149089,-3.90255878482325,0.967695543813140];    % denominator coefficients     
        case 1000
                b = [9.66139668268085e-11,3.86455867307234e-10,5.79683800960851e-10,3.86455867307234e-10,9.66139668268085e-11];
                a = [1,-3.98358125865852,5.95087842926670,-3.95101243657283,0.983715267510479];
        otherwise
                error('Filter Sampling Frequency Mismatch - Verify!!');
    end
    
  
    Yn = zeros(1,length(Xn));
    for i = 1:length(Yn)
        %Yn(i) =(1/a(1))*(b(1)*Xn(i) + b(2)*Xn_k(4) + b(3)*Xn_k(3) + b(4)*Xn_k(2) + b(5)*Xn_k(1)...
        %        - a(2)*Yn_k(4) - a(3)*Yn_k(3) - a(4)*Yn_k(2) - a(5)*Yn_k(1));
    Yn(i) = (b*[Xn(i) Xn_k(4) Xn_k(3) Xn_k(2) Xn_k(1)]' - a(2:5)*[Yn_k(4) Yn_k(3) Yn_k(2) Yn_k(1)]')/a(1);    
    Yn_k = [Yn_k(:,2:4) Yn(:,i)];
    Xn_k = [Xn_k(:,2:4) Xn(:,i)];
    
    end
