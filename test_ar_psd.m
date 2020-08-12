%  Calculate sliding and nonoverlapping window estimate of AR parameters
    k = 1;
    sig_samples  = 150; % 300ms window
    ar_coeff = [];
    while k < length(Diff_emg)
        if k+sig_samples-1 > length(Diff_emg)
            x_n = Diff_emg(1,k:end);
        else
            x_n = Diff_emg(1,k:k+sig_samples-1);
        end
        k = k+length(x_n); 
        [ar_a,ar_v,ar_k] = aryule(detrend(x_n),4);
        ar_coeff = [ar_coeff repmat(ar_a',1, length(x_n))];  
    end
    figure; plot(emgt,ar_coeff')
    hold on;plot(emgt,zscore(EMG_rms(1,:)),'k');   
    hold on; plot(emgt,EMG_rms(2,:),'g');
    