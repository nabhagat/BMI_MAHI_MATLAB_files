function Yn = pkt_bp_filter(Xn,Xn_k,Yn_k,samp_freq)
% Band Pass Filter EMG data packets with cutoff frequency of 30 - 200 Hz. This is a 4th order
% butterworth filter designed using fdatool. 
    
    % To verify that this filter is actually working with the EEG data: 
    % 1. Run the Brain Vision Recorder software in Simulated Amplifier mode. 
    % 2. Generate a single channel as a sine wave. 
    % 3. Save the processed_eeg and unprocessed_eeg variables from 
    %     the RDA client program. 
    % 4. Run the Inmotion_decoding program in closeloop testing mode. 
    % 5. Add some offset (~50 units) to the new_pkt variable which is
    %     received from brain vision. 
    % 6. View the proc_eeg & unproc_eeg offline to check if the filter
    %     removed the offset. 
    % 7. You can also use pmtm or pwelch.
    
    %[b,a] = butter(2,([30 200]./(samp_freq/2)),'bandpass')
    %% ***********Revisions 
%   5/15/2014 - Added switch statement to choose raw EEG sampling frequency. This affects the filter coefficients    
%   7/22/2014 - Modified from pkt_hp_filter()
%--------------------------------------------------------------------------------------------------
    switch samp_freq
        case 500
                b = [ 0.4808         0   -0.9617         0    0.4808];      % numerator coefficients
                a = [ 1.0000   -0.3457   -0.6317    0.0433    0.2523];    % denominator coefficients     
        case 1000
                b = [ 0.1600         0   -0.3200         0    0.1600];
                a = [ 1.0000   -2.2614    1.9845   -0.9277    0.2348];
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