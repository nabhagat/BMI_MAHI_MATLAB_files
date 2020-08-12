% Program to load Brain Vision files and extract EEG, EMG signals

% Created By:  Nikunj Bhagat, Graduate Student, University of Houston
% Contact : nbhagat08[at]gmail.com

%%
eeglab;
for block_num = 1:4
 
%  if block_num == 1   
%      EEG = pop_loadbv('C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_BNBO\BNBO_Session2\', ['BNBL_ses2_cond1_block000' num2str(block_num) '.vhdr'], [], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]);
%  else
     EEG = pop_loadbv('C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_BNBO\BNBO_Session2\', ['BNBO_ses2_cond1_block000' num2str(block_num) '.vhdr'], [], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]);
%  end
EEG.setname=['BNBO_ses2_cond1_block' num2str(block_num) '_eeg_raw'];
EEG = eeg_checkset( EEG );
EEG=pop_chanedit(EEG, 'lookup','C:\\Program Files\\MATLAB\\R2013a\\toolbox\\eeglab\\plugins\\dipfit2.2\\standard_BESA\\standard-10-5-cap385.elp');
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename',['BNBO_ses2_cond1_block' num2str(block_num) '_eeg_raw.set'],'filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
EEG = eeg_checkset( EEG );
% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%    eeglab redraw;

EEG = pop_select( EEG,'channel',{'TP9' 'TP10' 'FT9' 'FT10'});
EEG.setname=['BNBO_ses2_cond1_block' num2str(block_num) '_emg_raw'];
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename',['BNBO_ses2_cond1_block' num2str(block_num) '_emg_raw.set'],'filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
EEG = eeg_checkset( EEG );
% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%    eeglab redraw;

end



EEG = pop_mergeset( ALLEEG, [1  3  5 7], 0);
EEG.setname='BNBO_ses2_cond1_block80_eeg_raw';
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename','BNBO_ses2_cond1_block80_eeg_raw.set','filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
EEG = eeg_checkset( EEG );

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%    eeglab redraw;


EEG = pop_mergeset( ALLEEG, [2  4  6  8], 0);
EEG.setname='BNBO_ses2_cond1_block80_emg_raw';
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename','BNBO_ses2_cond1_block80_emg_raw.set','filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
EEG = eeg_checkset( EEG );

% Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
