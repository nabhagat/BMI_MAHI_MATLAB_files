function [ Unique_index,TrigOut] = ExtractUniqueTriggers( TrigIn,lookback,polarity)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% polarity = 1 -> detect +ve edge triggered signal (For InMotion)
%          = 0 -> detect -ve edge triggered signal (For Mahi-Exo)
Unique_index = [];
TrigOut = zeros(length(TrigIn),1);
if polarity == 1
    index = find(TrigIn == 1);
    for i = 1:length(index)
        if((i == 1) && (TrigIn(index(i))==1))
            Unique_index = [Unique_index; index(i)];
            TrigOut(index(i),1)=1;        
            continue
        end
        if lookback == 1
            if((TrigIn(index(i))==1) && (TrigIn(index(i)-1)==0))
            Unique_index = [Unique_index; index(i)];
            TrigOut(index(i),1)=1;
            end
        elseif lookback == 2
            if((TrigIn(index(i))==1) && (TrigIn(index(i)-1)==0) && (TrigIn(index(i)-2) == 0))
            Unique_index = [Unique_index; index(i)];
            TrigOut(index(i),1)=1;
            end
        end
    end

elseif polarity == 0
    
    index = find(TrigIn == 0);
    for i = 1:length(index)
%         if((i == 1) && (TrigIn(index(i))==5))
%             Unique_index = [Unique_index; index(i)];
%             TrigOut(index(i),1)=1;        
%             continue
%         end
        if lookback == 1
            if((TrigIn(index(i))==0) && (TrigIn(index(i)-1)==5))
            Unique_index = [Unique_index; index(i)];
            TrigOut(index(i),1)=1;
            end
%         elseif lookback == 2
%             if((TrigIn(index(i))==) && (TrigIn(index(i)-1)==0) && (TrigIn(index(i)-2) == 0))
%             Unique_index = [Unique_index; index(i)];
%             TrigOut(index(i),1)=1;
%             end
        end
    end
    
else

    error('Incorrect signal polarity');
end
    
    