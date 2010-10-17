% Performance of FAT, DMAC and SPT
clear,clc,close all,format compact
tic,
rand('twister', 8);
%set(0,'DefaultFigureWindowStyle','docked')

N=50; %Number of nodes
x = 200;
y = 200;
txRg = 60; % Radio transmission range
ixRg = 150; % Radio interference range

ac = 0;
nPkts = 10; % Number of packets
nTop = 1; % Number of topologies
vS = 8; % Vector with the number of sources

for k = 1:nTop
	fprintf('Topology = %d\n',k)
	%% Generate topology
	te1 = tree(N,[x y txRg ixRg]);
	for z = 1:length(vS) % number of sources
		% Generate the packets
		fprintf('Number of sources = %d\n',z)
 		te1 = genData(te1,vS(z),nPkts);
		te2 = prune(te1);
		[e,d,te3] = fat(te1,false);
		te4 = steiner(te1);
		te5 = center(te1);
		clf,subplot(1,2,1),plot(te4),subplot(1,2,2),plot(te5)
% 		for q = 1:2
% 			e(z,q) = e(z,q) + sum(engy(:,q)) / nTop;
% 			d(z,q) = d(z,q) + dly(q) / nTop;
% 		end
	end
end
toc
%%
% figure(1)
% plot(vS,e(:,1),vS,e(:,2))
% title('Energy consumption')
% ylabel('Energy (Joule)')
% xlabel('Number of sources')
% legend('dmac','RAGMAC')
% figure(2)
% plot(vS,d(:,1),vS,d(:,2))
% title('Delay')
% legend('dmac','RAGMAC')