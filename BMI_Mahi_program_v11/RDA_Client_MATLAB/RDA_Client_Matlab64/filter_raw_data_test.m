function filtered_data = filter_raw_data_test(new_data,prev_data)

    window_length = 5;
    append_data = [prev_data(:,(length(prev_data)-(window_length-1)+1):length(prev_data)) new_data(:,:)];
    filtered_data = zeros(1,length(new_data));
    for fil_ind = window_length:length(append_data)
        filtered_data(fil_ind) = (1/window_length)*(append_data(fil_ind) + append_data(fil_ind-1) + append_data(fil_ind-2) + append_data(fil_ind-3) + append_data(fil_ind-4));
    end

filtered_data = filtered_data(:,5:length(append_data));