function Yn = pkt_hp_filter(Xn,Xn_k,Yn_k,samp_freq)
% High Pass Filter EEG data packets with cutoff frequency of 0.1 Hz. This is a 4th order
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
    
    %[b,a] = butter(4,(0.1/(samp_freq/2)),'high')
    %% ***********Revisions 
%   5/15/2014 - Added switch statement to choose raw EEG sampling frequency. This affects the filter coefficients    

%--------------------------------------------------------------------------------------------------
    switch samp_freq
        case 500
                b = [0.998359471568987,-3.99343788627595,5.99015682941392,-3.99343788627595,0.998359471568987];      % numerator coefficients
                %b = [0.967694808889671,-3.87077923555868,5.80616885333802,-3.87077923555868,0.967694808889671;];
                a = [1,-3.99671624921533,5.99015413808164,-3.99015952333533,0.996721634471508];    % denominator coefficients     
                %a = [1,-3.93432582079874,5.80512542105513,-3.80723245722884,0.936433243152017;];
        case 1000
                b = [0.999179399138992,-3.99671759655597,5.99507639483395,-3.99671759655597,0.999179399138992];
                a = [1,-3.99835812456834,5.99507572144826,-3.99507706854351,0.998359471663757];
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