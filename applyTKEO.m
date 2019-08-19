function signal_out = applyTKEO(signal_in)
    % Based on 
    % https://www.mathworks.com/matlabcentral/fileexchange/59347-emg-processing-and-action-potential-detection-with-multi-resolution-teager-operator
    % calcs the function x(n)^2 - x(n-1)*x(n+1)
    [r,c] = size(signal_in);    
    signal_out = signal_in.^2 - ([zeros(r,1),signal_in(:,1:end-1)].*[signal_in(:,2:end),zeros(r,1)]);
end