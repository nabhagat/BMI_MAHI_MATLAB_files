function [ procEEGdata ] = spatial_filter(rawEEGdata,filter_type,eliminate_chns_from_ref, emg_used)
% spatial_filter() uses different spatial filters for preprocessing of EEG signals 
% By Nikunj Bhagat, University of Houston
% Detailed explanation goes here

if isempty(rawEEGdata)
    error('Raw EEG data required');
end

[total_chnns,ncol] = size(rawEEGdata);      %rawEEGdata = [channels x samples]
Spatial_matrix = eye(total_chnns);

switch filter_type
    case 'CAR'
%% Create Common Average Reference Matrix
        num_car_channels = total_chnns - length(eliminate_chns_from_ref);          % Number of channels to use for CAR
        Spatial_matrix =  (1/num_car_channels)*(diag((num_car_channels-1)*ones(total_chnns,1)) - (ones(total_chnns,total_chnns)-diag(ones(total_chnns,1))));
        
        if ~isempty(eliminate_chns_from_ref)
            for elim = 1:length(eliminate_chns_from_ref)
                Spatial_matrix(:,eliminate_chns_from_ref(elim)) = 0;
            end
        end     
    case 'LLAP'
%% Create Large Laplacian Matrix

if (emg_used)
    Neighbors = [
             % Row F
             37,1, 3, 47, 38; 
             4,  1, 3, 13, 5;
            38, 1, 37, 48, 39;
             5, 28, 4, 14, 6;       
            39, 2, 38, 49, 40;
             6,  2, 5, 15, 7;
             40, 2, 39, 50, 7; 
             
             % Row FC
             % 8, 33, 41, 18, 9;      % Redefined on 7-11-2017. Instead of Ch. 41 use Ch. 12 (T7)
             8, 33, 12, 18, 9;
             43, 34, 8, 52, 32;        % Instead of Ch. 42 using  Ch. 8, because Ch. 42 is now being used for recording EMG - 8/28/2015
             %43, 34, 42, 52, 32;        % Reverting from above
              9, 34, 8, 19, 10;
             32, 28, 43, 53, 44;    
             10, 35, 9, 20, 11;
             44, 35, 32, 54, 11;    % Instead of Ch. 45 using  Ch. 11
             %44, 35, 32, 54, 45;        % Reverting from above
             %11, 36, 10, 21, 46;
             11, 36, 10, 21, 16;        % Redefined on 7-11-2017. Instead of Ch. 46 use Ch. 16 (T8)
             
             % Row C
             47, 37, 12, 56, 48; 
             13, 4, 12, 24, 14;
             48, 38, 47, 57, 49;
             14, 5, 13, 25, 15;
             49, 39, 48, 58, 50;
             15, 6, 14, 26, 16;
             50, 40, 49, 59, 16;
             
             % Row CP
             %18, 8, 17, 60, 19;    % Redefined on 7-11-2017. Instead of Ch. 17 use Ch. 12 (T7)
             18, 8, 12, 60, 19;
             52, 43, 18, 60, 53;    % Instead of Ch. 51 using  Ch. 18
             %52, 43, 51, 60, 53;        % Reverting from above
             19, 9, 18, 61, 20; 
             53, 32, 52, 62, 54;
             20, 10, 19, 63, 21;
             54, 44, 53, 64, 21;    % Instead of Ch. 55 using  Ch. 21
             %54, 44, 53, 64, 55;        % Reverting from above
             %21, 11, 20, 64, 22;   % Redefined on 7-11-2017. Instead of Ch. 22 use Ch. 16 (T8)
             21, 11, 20, 64, 16;
             
             % Row P
             56, 47, 23, 29, 57;  
             24, 13, 23, 29, 25;
             57, 48, 56, 30, 58;
             25, 14, 24, 30, 26;
             58, 49, 57, 30, 59;
             26, 15, 25, 31, 27;
             59, 50, 58, 31, 27];
else % Force sensor used
    Neighbors = [                                         
             % Row F
             37,1, 3, 47, 38; 
              4,  1, 3, 13, 5;
             38, 1, 37, 48, 39;
              5, 28, 4, 14, 6;       
             39, 2, 38, 49, 40;
              6,  2, 5, 15, 7;
             40, 2, 39, 50, 7; 

              % Row FC
             8, 33, 41, 18, 9;      % Redefined on 7-11-2017. Use Ch. 41 because EMG not used                                                                                  
             43, 34, 42, 52, 32;    % Reverting because EMG not used
              9, 34, 8, 19, 10;
             32, 28, 43, 53, 44;    
             10, 35, 9, 20, 11;                                         
             44, 35, 32, 54, 45;    % Reverting because EMG not used
             11, 36, 10, 21, 46;    % Redefined on 7-11-2017. Use Ch. 46 because EMG not used

             % Row C
             47, 37, 12, 56, 48; 
             13, 4, 12, 24, 14;
             48, 38, 47, 57, 49;
             14, 5, 13, 25, 15;
             49, 39, 48, 58, 50;
             15, 6, 14, 26, 16;
             50, 40, 49, 59, 16;

             %18, 8, 17, 60, 19;    % Redefined on 7-11-2017. Instead of Ch. 17 use Ch. 12 (T7)
             18, 8, 12, 60, 19;     % because Ch. 17 is used for recording force                                         
             52, 43, 51, 60, 53;    % Reverting because EMG not used
             19, 9, 18, 61, 20; 
             53, 32, 52, 62, 54;
             20, 10, 19, 63, 21;                                         
             54, 44, 53, 64, 55;    % Reverting because EMG not used
             21, 11, 20, 64, 22;   % Redefined on 7-11-2017. Use Ch. 22 because EMG not used                                          

             % Row P
             56, 47, 23, 29, 57;  
             24, 13, 23, 29, 25;
             57, 48, 56, 30, 58;
             25, 14, 24, 30, 26;
             58, 49, 57, 30, 59;
             26, 15, 25, 31, 27;
             59, 50, 58, 31, 27];
end
            for nn_row = 1:size(Neighbors,1) 
                Spatial_matrix(Neighbors(nn_row,1),Neighbors(nn_row,2:end)) = -0.25;
            end
        
    case 'SLAP'
%% Create Small Laplacian Matrix
Neighbors = [
             % Row F
             4, 34, 37, 43, 38;
            38, 34, 4, 9, 5;
             5, 28, 38, 32, 39;
            39, 35, 5, 10, 6;
             6, 35, 39, 44, 40;
             % Row FC
             43, 4, 8, 13, 9;
              9, 38, 43, 48, 32;
             32, 5, 9, 14, 10;        
             10, 39, 32, 49, 44;       
             44, 6, 10, 15, 11;       
             % Row C
             13, 43, 47, 52, 48;
             48, 9, 13, 14, 19;
             14, 32, 48, 53, 49;
             49, 10, 14, 20, 15;
             15, 44, 49, 54, 50;
             % Row CP
             52, 13, 18, 24, 19;
             19, 48, 52, 57, 53; 
             53, 14, 19, 25, 20;
             20, 49, 53, 58, 54;
             54, 15, 20, 26, 21;
             % Row P
             24, 52, 56, 61, 57;
             57, 19, 24, 61, 25;
             25, 53, 57, 62, 58;
             58, 20, 25, 63, 26;
             26, 54, 58, 63, 59;];
         
            for nn_row = 1:size(Neighbors,1) 
                Spatial_matrix(Neighbors(nn_row,1),Neighbors(nn_row,2:end)) = -0.25;
            end
                     
    case 'WAVG'
%% Create Weigthed Average Matrix
Neighbors = [
             % Row F
             4, 34, 37, 43, 38;
            38, 34, 4, 9, 5;
             5, 28, 38, 32, 39;
            39, 35, 5, 10, 6;
             6, 35, 39, 44, 40;
             % Row FC
             43, 4, 8, 13, 9;
              9, 38, 43, 48, 32;
             32, 5, 9, 14, 10;        
             10, 39, 32, 49, 44;       
             44, 6, 10, 15, 11;       
             % Row C
             13, 43, 47, 52, 48;
             48, 9, 13, 14, 19;
             14, 32, 48, 53, 49;
             49, 10, 14, 20, 15;
             15, 44, 49, 54, 50;
             % Row CP
             52, 13, 18, 24, 19;
             19, 48, 52, 57, 53; 
             53, 14, 19, 25, 20;
             20, 49, 53, 58, 54;
             54, 15, 20, 26, 21;
             % Row P
             24, 52, 56, 61, 57;
             57, 19, 24, 61, 25;
             25, 53, 57, 62, 58;
             58, 20, 25, 63, 26;
             26, 54, 58, 63, 59;];
         
            for nn_row = 1:size(Neighbors,1) 
                Spatial_matrix(Neighbors(nn_row,1),Neighbors(nn_row,2:end)) = 0.25; % Anti-Laplcian
            end
        
    otherwise
        error('Invalid Spatial Filter Type.');
end

   procEEGdata = Spatial_matrix*(rawEEGdata);
end

