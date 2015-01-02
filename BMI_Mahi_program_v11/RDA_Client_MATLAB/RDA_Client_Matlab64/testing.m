data1s = [];
                        data500ms = [];
                        prev_pkt = zeros(1,(1000000 / props.samplingInterval)/2);
                        new_pkt  = zeros(1,(1000000 / props.samplingInterval)/2);
                        processed_eeg = [];
                        unprocessed_eeg = [];


data500ms = rand(1,255);
dims = size(data500ms);

if dims(2) > (1000000 / props.samplingInterval)/2
%                             data1s = data1s(:, dims(2) - 1000000 / props.samplingInterval : dims(2));
%                             avg = mean(mean(data1s.*data1s));
%                             disp(['Average power: ' num2str(avg)]);
                            % set data buffer to empty for next full second
%                            data1s = [];
                            new_pkt = data500ms(1:(1000000 / props.samplingInterval)/2);
                            data500ms = data500ms(:,((1000000 / props.samplingInterval)/2)+1 : dims(2));
                            %data500ms = [];
                            fil_eeg = filter_raw_data_test(new_pkt,prev_pkt);
                            prev_pkt = new_pkt;
                            processed_eeg = [processed_eeg fil_eeg];
                            unprocessed_eeg = [unprocessed_eeg prev_pkt];
end

%                 

