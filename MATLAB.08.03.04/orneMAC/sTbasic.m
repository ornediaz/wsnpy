clear,clc, close all,format compact
tic
set(0,'DefaultFigureWindowStyle','docked')
%% Create topology
N=20;
BOwn = cell(1,N);
xSide = 200;
ySide = 20;
txRg = 60; % Radio transmission range
ixRg = 150; % Radio interference range

[tier,upTierID,interfRng,trnRng,P,previous] = ...
	topology(N,txRg,ixRg,xSide,ySide);
figure
plt(P,previous,xSide,ySide)

for z = [5 8 9 4 10 12]  % Name of the node that originates the traffic
	for gg = 1:2; % Packet number
		% Append the packet in the output buffer
		BOwn{z} = [BOwn{z} newPkt(z,upTierID{z}(1),DataT,'DataK',gg,z)];
	end
end
%% Compute
engy = zeros(N,2);
dly = zeros(1,2);
[engy(:,1),dly(1)] = dmac(N,tier,upTierID,trnRng,interfRng,BOwn,false);
[engy(:,2),dly(2)] = orneMAC(...
	N,tier,upTierID,trnRng,interfRng,BOwn,P,xSide,ySide,false);
fprintf('DMAC:: engy = %7.4f, delay = %7.4f\n',sum(engy(:,1)),dly(1));
fprintf('OMAC:: engy = %7.4f, delay = %7.4f\n',sum(engy(:,2)),dly(2));
%% Display results
figure(3)
plot(1:N,[engy(:,1) engy(:,2)])
legend('DMAC','OMAC')
toc
