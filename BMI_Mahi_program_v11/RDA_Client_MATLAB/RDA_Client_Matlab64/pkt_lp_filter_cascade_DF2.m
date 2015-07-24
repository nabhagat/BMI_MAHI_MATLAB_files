function [Yn,params,Yn1] = pkt_lp_filter_cascade_DF2(Xn,params,samp_freq)
% Low Pass Filter EEG data packets with cutoff frequency of 1 Hz. This is a 4th order
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
    
    %[b,a] = butter(4,(0.1/(samp_freq/2)),'low')
    %% ***********Revisions 
%   5/15/2014 - Added switch statement to choose raw EEG sampling frequency. This affects the filter coefficients    
%   7/1/2015 - Filter Implementation modified to cascade of two second-order filters.
%                      - params = [w(n-1) w(n-2); v(n-1) v(n-2)]; w's are
%                      delays coefficents for first sos H1(z) and v's are
%                      delays for H2(z)

% Structure of SOS matrix
                % [b10 b11 b12 1 a11 a12;       --> H1(z)
                %  b20 b21 b22 1 a21 a22];      --> H2(z)
%--------------------------------------------------------------------------------------------------
    switch samp_freq
        case 500
                %b = [0.998359471568987,-3.99343788627595,5.99015682941392,-3.99343788627595,0.998359471568987];      % numerator coefficients
                %a = [1,-3.99671624921533,5.99015413808164,-3.99015952333533,0.996721634471508];    % denominator coefficients     
                
                % SOS_matrix = tf2sos(b,a)
                              
                % HPF - unstable filter
                %SOS_matrix = [ 0.9984   -1.9967    0.9984    1.0000   -1.9977    0.9977;
                %                            1.0000   -2.0000    1.0000    1.0000   -1.9990    0.9990];
                
                %LPF
                 SOS_matrix = [0.000000001533245   0.000000003066491   0.000000001533246   1.000000000000000 -1.976891354671267   0.977047454018255;
                                            1.000000000000000   1.999999938672026   0.999999985818274   1.000000000000000 -1.990271241277580   0.990428397140121];

                                        
        case 1000
                %b = [0.999179399138992,-3.99671759655597,5.99507639483395,-3.99671759655597,0.999179399138992];
                %a = [1,-3.99835812456834,5.99507572144826,-3.99507706854351,0.998359471663757];
                
                % HPF - unstable filter
                % SOS_matrix = [ 0.9992   -1.9984    0.9992    1.0000   -1.9988    0.9988;
                %                            1.0000   -2.0000    1.0000    1.0000   -1.9996    0.9996];
                
                % LPF
                SOS_matrix = [ 0.999179399138992  -1.998358892601164   0.999179403071598   1.000000000000000  -1.998796874965740 0.998797317702321;
                                            1.000000000000000  -1.999999905599362   0.999999996064167   1.000000000000000  -1.999561249602604 0.999561626737677];  
                                        
        otherwise
                error('Filter Sampling Frequency Mismatch - Verify!!');
    end

    
    Yn = zeros(1,length(Xn));
    Yn1 = zeros(1,length(Xn));
    for i = 1:length(Xn)
        %Yn(i) =(1/a(1))*(b(1)*Xn(i) + b(2)*Xn_k(4) + b(3)*Xn_k(3) + b(4)*Xn_k(2) + b(5)*Xn_k(1)...
        %        - a(2)*Yn_k(4) - a(3)*Yn_k(3) - a(4)*Yn_k(2) - a(5)*Yn_k(1));
    %Yn(i) = (b*[Xn(i) Xn_k(4) Xn_k(3) Xn_k(2) Xn_k(1)]' - a(2:5)*[Yn_k(4) Yn_k(3) Yn_k(2) Yn_k(1)]')/a(1);    
    %Yn_k = [Yn_k(:,2:4) Yn(:,i)];
    %Xn_k = [Xn_k(:,2:4) Xn(:,i)];
    
    % Cascade filter realization
    % H1(z)
    Wn = SOS_matrix(1,4:6)*[Xn(i) -1.*params(1,1) -1.*params(1,2)]';
    Yn1(i) = SOS_matrix(1,1:3)*[Wn params(1,1) params(1,2)]';       % Y1(n) - output of H1(z)
    params(1,2) = params(1,1);
    params(1,1) = Wn;
    
    Vn = SOS_matrix(2,4:6)*[Yn1(i) -params(2,1) -params(2,2)]';
    Yn(i) = SOS_matrix(2,1:3)*[Vn params(2,1) params(2,2)]';
    params(2,2) = params(2,1);
    params(2,1) = Vn;   
    
    end