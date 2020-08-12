
% %% Test for extracting events sub-types
% 
% for event_cnt = 1:length(EEG.event)
%     if strcmp(EEG.event(event_cnt).type,'square')
%         switch EEG.event(event_cnt).position
%             case 1
%                 EEG.event(event_cnt).type = 'square1';
%                 
%             case 2 
%                 EEG.event(event_cnt).type = 'square2';
%                 
%             otherwise
%                 warning('Error in switch');
%         end
%     end
% end


%% Assign target numbers to events, 14-8-2013
% Use NB_downsample_lpf.set

% 0 - Center
% 1 - South
% 2 - West
% 3 - North
% 4 - East
% 5 - Rest

target_cnt = 1;
for event_cnt = 3:length(EEG.event)         % Ignore boundary + start triggers
    if strcmp(EEG.event(event_cnt).type,'R128')
        if target_cnt <= length(targets_full);        
            switch targets_full(target_cnt)
                case 0
                    EEG.event(event_cnt).type = 'center';
                    target_cnt = target_cnt + 1;
                case 1
                    EEG.event(event_cnt).type = 'south';
                    target_cnt = target_cnt + 1;
                    
                case 3
                    EEG.event(event_cnt).type = 'north';
                    target_cnt = target_cnt + 1;
                    
                case 5
                    EEG.event(event_cnt).type = 'rest';
                    target_cnt = target_cnt + 1;

                otherwise warning('Error in switch');
            end
        end
    end
end