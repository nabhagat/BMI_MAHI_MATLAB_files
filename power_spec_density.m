% Power Spectrum Density usin DFT/FFT
% Effect of length (L) of signal (x_n) on the PSD. The importance of large L irrespective of large Nfft
% For 2 nearby frequencies to be resolved properly, (omega1 - omega2) >= (2pi/L) , where omega = digital freq in rad
% The fft length (Nfft) only calculates more points in the PSD. It does not further resolve the individual frequencies

clear;

%x_n = [0 1 2 3];
%X_k = fft(x_n,10);
%disp(X_k);

Fs = 200; 
t = 0:1/Fs:10 - 1/Fs;
F1 = 5;  F2 = 8; 
x_t = cos(2*pi*F1*t) + cos(2*pi*F2*t);

L1 = 20; % length of recorded signal
x_n1 = x_t(1:L1);
Nfft = 2048;
X_k1 = fft(x_n1,Nfft);      % check X_k(1) = sum(x_n)
X_k_magn1 = abs(X_k1);      
freq1 = 0:Fs/Nfft:Fs/2;
figure; 
%plot(freq1,X_k_magn1(1:length(freq1)),'LineWidth',2,'k');

L2 = 100; % length of recorded signal
x_n2 = x_t(1:L2);
Nfft = 2048;
X_k2 = fft(x_n2,Nfft);      % check X_k(1) = sum(x_n)
X_k_magn2 = abs(X_k2);      
freq2 = 0:Fs/Nfft:Fs/2;
hold on; 
%plot(freq2,X_k_magn2(1:length(freq2)),'LineWidth',2,'r');

L3 = 150; % length of recorded signal
x_n3 = x_t(1:L3);
Nfft = 2048;
X_k3 = fft(x_n3,Nfft);      % check X_k(1) = sum(x_n)
%X_k_magn3 = abs(X_k3);      
X_k_magn3 = (abs(X_k3).^2)/L3;  % Periodogram
freq3 = 0:Fs/Nfft:Fs/2;
hold on; 
plot(freq3,X_k_magn3(1:length(freq3)),'LineWidth',2,'k');

pkg load signal
%% Power Spectrum Estimation using Auto-correlation
[Rxx,lags] = xcorr(x_n3,'biased');
Nfft = 2048;
Pxx = abs(fft(Rxx,Nfft));      % Why take absolute value? Does not match with formula in text-book
Pxx_freq = 0:Fs/Nfft:Fs/2;
hold on; 
plot(Pxx_freq,Pxx(1:length(Pxx_freq)),'LineWidth',2,'r');

%% Parametric methods for Spectrum Estimation
model_order = 50;
figure
%1. Yule-Walker Method
[yule_a,yule_v,yule_k] = aryule(x_n3,model_order);
[yule_psd,yule_freq] = ar_psd(yule_a,yule_v,1025,Fs);
hold on;
plot(yule_freq,yule_psd,'LineWidth',2,'b');

%2. Burg Method
[burg_a,burg_v,burg_k] = arburg(detrend(x_n3),model_order,'AKICc');
[burg_psd,burg_freq] = ar_psd(burg_a,burg_v,1025,Fs);
hold on;
plot(burg_freq,burg_psd,'LineWidth',2,'m');
legend('Yule-Walker AR', 'Burg AR');


%% Model order selection for parametric method using partial autocorrelation
figure
pacf = -burg_k;          % The negative of reflection coeff is the partial autocorrelation sequence
lag = 1:length(burg_a)-1;
stem(lag,pacf,'markerfacecolor',[0 0 1]);
xlabel('Lag'); ylabel('Partial Autocorrelation');
set(gca,'xtick',1:1:15)
lconf = -1.96/sqrt(1000)*ones(length(lag),1);
uconf = 1.96/sqrt(1000)*ones(length(lag),1);
hold on;
line(lag,lconf,'color',[1 0 0],'LineWidth',2);
line(lag,uconf,'color',[1 0 0],'LineWidth',2);

