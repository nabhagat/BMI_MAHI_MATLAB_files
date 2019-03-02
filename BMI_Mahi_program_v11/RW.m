    % obj = serial('com3','baudrate',9600,'parity','none','databits',8,'stopbits',1);
% fopen(obj);
% fwrite(obj,'a');
% a = fread(obj, 3, 'uchar');
% fprintf('%s\n', a);
% fclose(obj);
% delete(obj);
% clear obj;

obj = serial('com44','baudrate',19200,'parity','none','databits',8,'stopbits',1);
fopen(obj);
%a = fread(obj, 1, 'uchar');              % No longer used. Nikunj09/05/2014
%fprintf('%s\n', a);
%if a == 'm'
fwrite(obj,7);          % 7 - Movement onset
%end
fclose(obj);
delete(obj);
clear obj;