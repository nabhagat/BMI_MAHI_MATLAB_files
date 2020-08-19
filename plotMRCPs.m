function plotMRCPs(Data,Subject_name, MRCP_channels,optimized_MRCP_channels)
%UNTITLED Summary of this function goes here
%   Function to plot Grand Averaged MRCPs and highlight classification
%   channels
    
    Channel_nos = [ 4, 38,   5,  39,   6,... 
                     43,  9,  32, 10,  44,...
                     13, 48,  14, 49,  15,...
                     52, 19,  53, 20,  54,... 
                     24, 57,  25, 58,  26];  
                 
    Channel_labels = {'F_{3}','F_{1}','F_{Z}','F_{2}','F_{4}',...
                      'FC_{3}','FC_{1}','FC_{Z}','FC_{2}','FC_{4}',...
                      'C_{3}','C_{1}','C_{Z}','C_{2}','C_{4}',...
                      'CP_{3}','CP_{1}','CP_{Z}','CP_{2}','CP_{4}',...
                      'P_{3}','P_{1}','P_{Z}','P_{2}','P_{4}'};
    figure('NumberTitle', 'off', 'Name', Subject_name);
    %figure('units','normalized','outerposition',[0 0 1 1])
    T_plot = tight_subplot(numel(Channel_nos)/5,5,[0.02 0.02],[0.15 0.01],[0.1 0.1]);
    hold on;        

    for ind4 = 1:length(Channel_nos)
        axes(T_plot(ind4));    
        hold on;
        m1 = plot(Data.move_erp_time,Data.move_avg_channels(Channel_nos(ind4),:),'k','LineWidth',1);
%         plot(Data.move_erp_time,Data.move_avg_channels(Channels_nos(ind4),:)+ (Data.move_SE_channels(Channels_nos(ind4),:)),'--','Color',[0 0 0],'LineWidth',0.25);
%         plot(Data.move_erp_time,Data.move_avg_channels(Channels_nos(ind4),:) - (Data.move_SE_channels(Channels_nos(ind4),:)),'--','Color',[0 0 0],'LineWidth',0.25);

        r1 = plot(Data.rest_erp_time,Data.rest_avg_channels(Channel_nos(ind4),:),'Color',[0.6 0.6 0.6],'LineWidth',1,'LineStyle','-');
%         plot(Data.rest_erp_time,Data.rest_avg_channels(Channels_nos(ind4),:)+ (Data.rest_SE_channels(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);
%         plot(Data.rest_erp_time,Data.rest_avg_channels(Channels_nos(ind4),:)+ (Data.rest_SE_channels(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);

        % Added 8/21/2015
        if (strcmp(Subject_name,'S9023') || strcmp(Subject_name,'S9007'))  
            text(-2,-4,Channel_labels(ind4),'Color','k','FontWeight','normal','FontSize',8); 
        else
            text(-2,-3,Channel_labels(ind4),'Color','k','FontWeight','normal','FontSize',8); 
        end
        set(gca,'YDir','reverse');
        if (strcmp(Subject_name,'S9023') || strcmp(Subject_name,'S9007'))
            axis([-2.5 1 -15 10]);                
        else
            axis([-2.5 1 -5 5]);                
        end                    
        line([0 0],[-30 20],'Color','k','LineWidth',0.5,'LineStyle','--');  
        line([-2.5 4],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
%             plot_ind4 = plot_ind4 + 1;
        grid off;
        if ~isempty(find(Channel_nos(ind4) == MRCP_channels))
            if (strcmp(Subject_name,'S9023') || strcmp(Subject_name,'S9007'))       
                rectangle('Position',[-2.4 -14 3.3 23],...                      
                          'EdgeColor', 'b',...
                          'LineWidth', 1,...
                          'LineStyle','-');
            else
                rectangle('Position',[-2.4 -4.6 3.3 9.2],...                      
                          'EdgeColor', 'b',...
                          'LineWidth', 1,...
                          'LineStyle','-');
            end
        end
        
        if ~isempty(find(Channel_nos(ind4) == optimized_MRCP_channels))
            if (strcmp(Subject_name,'S9023') || strcmp(Subject_name,'S9007'))
                rectangle('Position',[-2.5 -15 3.5 25],...                      
                          'EdgeColor', 'r',...
                          'LineWidth', 1,...
                          'LineStyle','-');
            else
                rectangle('Position',[-2.4 -5 3.4 10],...                      
                          'EdgeColor', 'r',...
                          'LineWidth', 1,...
                          'LineStyle','-');
            end                      
        end
        
        if (Channel_nos(ind4) == 24)
            set(gca,'XTick',[-2 -1 0 1], 'XTickLabel',{'-2', '-1', '0', '1'});
            if (strcmp(Subject_name,'S9023') || strcmp(Subject_name,'S9007'))
                set(gca,'YTick',[-10 0 10], 'YTickLabel',{'-10', '0', '10'});
            else
                set(gca,'YTick',[-5 0 5], 'YTickLabel',{'-5', '0', '5'});
            end
            xlabel('Time (s)'); ylabel('Voltage (\muV)');            
            legend([m1 r1],{'MRCP','Avg. resting EEG'},'Location','southoutside','Orientation','horizontal');
        else
%             set(gca,'Visible', 'off');
%             set(gca,'XTick',[-2 -1 0 1], 'XTickLabel',{''});
%             set(gca,'YTick',[-10 0 10], 'YTickLabel',{''});              
        end
        
    end
    disp([Subject_name ', electrodes retained = ', num2str(length(optimized_MRCP_channels)/length(MRCP_channels)) '%']);
end

