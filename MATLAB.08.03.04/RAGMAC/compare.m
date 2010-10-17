% Performance of FAT, DMAC and SPT
clear,clc,close all,format compact
tic,
rand('twister', 9);
%set(0,'DefaultFigureWindowStyle','docked')

N = 50; %Number of nodes
x = 200;
y = 200;
txRg = 60; % Radio transmission range
ixRg = 150; % Radio interference range

acV = linspace(0,1,4); %aggregation coefficient vector
nPkts = 10; % Number of packets
nTop = 10; % Number of topologies
vS = 2:2:12; % Vector with the number of sources

s = zeros(6,2,length(vS),length(acV));
for k = 1:nTop
	fprintf('Topology = %d\n',k)
	%% Generate topology
	te1 = tree(N,[x y txRg ixRg]);
	for z = 1:length(vS) % number of sources
		for ac = 1:length(acV)
			% Generate the packets
			fprintf('Number of sources = %d\n',vS(z))
			te1 = genData(te1,vS(z),nPkts);
			saveState
			[e,d] = dmac(te1,false);
			s(1,:,z,ac) = s(1,:,z,ac) + [sum(e) d]/nTop;
			saveState
			te2 = prune(te1);
			[e,d] = txPhase(te2,acV(ac),false);
			s(2,:,z,ac) = s(2,:,z,ac) + [sum(e) d]/nTop;
			saveState
			[e,d,te3] = fat(te1,false);
			s(3,:,z,ac) = s(3,:,z,ac) + [sum(e) d]/nTop;
			saveState
			[e,d] = txPhase(te3,acV(ac),false);
			s(4,:,z,ac) = s(4,:,z,ac) + [sum(e) d]/nTop;
			saveState
			te4 = steiner(te1);
			[e,d] = txPhase(te4,acV(ac),false);
			s(5,:,z,ac) = s(5,:,z,ac) + [sum(e) d]/nTop;
			saveState
			te5 = center(te1);
			[e,d] = txPhase(te5,acV(ac),false);
			s(6,:,z,ac) = s(6,:,z,ac) + [sum(e) d]/nTop;
			clf,subplot(1,2,1),plot(te3),subplot(1,2,2),plot(te5)
		end
	end
end
disp(s)
save data2
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
load data2
close all,
set(0,'DefaultAxesFontName','Times New Roman')
set(0,'DefaultAxesFontSize',9)
set(0,'DefaultAxesColorOrder',[0 0 0],...
	'DefaultAxesLineStyleOrder','-+|-.^|--v|:o|:d|:<|:>')
figure(1)
plot(acV,squeeze(s([1 2 3 4 6],2,end,:))')
legend('DMAC','SPT','FAT-setup','FAT-tx','Steiner2',...
	'Location','SouthEast')
xlabel('Aggregation coefficient \alpha')
ylabel('Delay (second)')
%title('nPkts = 10,x = y = 200, txRg = 60,ixRg = 150, N = 50')
axis([-.1 1 0 7])
impMeta('graph1',[9 6])



figure(2)
plot(vS,squeeze(s([1 2 3 4 6],2,:,1))')
legend('DMAC','SPT','FAT-setup','FAT-tx','Steiner2','Location','NorthWest')
xlabel('Number of sources')
ylabel('Delay (second)')
axis([2 12 0 5])
impMeta('graph2',[9 6])