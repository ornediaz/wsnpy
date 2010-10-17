% Plot the data provided by Holroyd and Chorus
clear,clc,close all
set(0,'DefaultFigureWindowStyle','docked')
load AE

figure(1)
plot(data1(:,1),data1(:,2))
title('Lab test -2 rubber pads. Wire cut #1 (~90 KHz)')
xlabel('time us')
ylabel('Value (mV)')

figure(2)
plot(data2(:,1),data2(:,2))
title('Lab test -2 rubber pads. Wire cut #2 (~90 KHz)')
xlabel('time us')
ylabel('Value (mV)')


figure(3)
for kk=1:4
	subplot(2,2,kk)
	plot(data3(:,1+(kk-1)*2),data3(:,2+(kk-1)*2))
end

figure(4)
plot(data4(:,1),data4(:,2))
title('Humber Bridge main cable. Impact at 5m (~40 KHz)')
xlabel('time (ms)')
ylabel('voltage (mV)')

figure(5)
plot(abs(fft(data1(:,2))))
title('Spectrum of the captured signal. Decimation 4:1 possible')

