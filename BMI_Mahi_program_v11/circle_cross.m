function handle_circle_cross = circle_cross(current_axes_handle,center_x, center_y, Radius,width,color)
% Function to create circle with cross for Flexion_Game GUI
% Created by: Nikunj A. Bhagat, University of Houston
% Date: July 10th, 2014

axes_xlim = get(current_axes_handle,'XLim');
axes_ylim = get(current_axes_handle,'YLim');

outer_x = center_x - Radius - axes_xlim(1);
outer_y = center_y - Radius - axes_ylim(1);
outer_w = 2*Radius;
outer_h = 2*Radius;
outer_circle = rectangle('Position',[outer_x outer_y outer_w outer_h],...
                         'Curvature',[1 1], 'FaceColor',color,'Visible','off');

inner_x = outer_x + width;
inner_y = outer_y + width;
inner_w = 2*(Radius - width);
inner_h = 2*(Radius - width);
inner_circle = rectangle('Position',[inner_x inner_y inner_w inner_h],...
                         'Curvature',[1 1], 'FaceColor',[1 1 1],'Visible','off');

vertical_x = center_x - width/2;
vertical_y = center_y - Radius + width/2;
vertical_w = width;
vertical_h = 2*Radius - width;
vertical_bar = rectangle('Position',[vertical_x vertical_y vertical_w vertical_h],...
                         'Curvature',[0 0], 'FaceColor',color,'Visible','off');
                     
horizontal_x = center_x - Radius + width/2;
horizontal_y = center_y - width/2;
horizontal_w = 2*Radius - width;
horizontal_h = width;
horizontal_bar = rectangle('Position',[horizontal_x horizontal_y horizontal_w horizontal_h],...
                         'Curvature',[0 0], 'FaceColor',color,'Visible','off');
                     
handle_circle_cross = [outer_circle inner_circle vertical_bar horizontal_bar];                     