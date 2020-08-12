% EMG_ICA

figure;
subplot(4,1,1);
plot(EEG.data(1,:));
subplot(4,1,2);
plot(EEG.data(2,:),'r');
subplot(4,1,3);
plot(EEG.data(3,:));
subplot(4,1,4);
plot(EEG.data(4,:),'r');

EEGsources = EEG.icawinv*EEG.data;

figure;
subplot(4,1,1);
plot(EEGsources(1,:));
subplot(4,1,2);
plot(EEGsources(2,:),'r');
subplot(4,1,3);
plot(EEGsources(3,:));
subplot(4,1,4);
plot(EEGsources(4,:),'r');
