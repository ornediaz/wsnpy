clear,clc, close all,format compact
tic,
rand('twister', 5489);
set(0,'DefaultFigureWindowStyle','docked')
N=20; %Number of nodes
BOwn = cell(1,N);
x = 200;
y = 50;
tx = 60; % Radio transmission range
ix = 150; % Radio interference range
nPkts = 8; % Number of packets
nTop = 1; % Number of topologies
vS = 5 : 5:10; % Vector with the number of sources
e = zeros(length(vS),2); %energy consumed
d = e; % delay
for k = 1: nTop
	fprintf('Topology = %d\n',k)
	toc
	[T,upT,Tx,Ix,P,p] = topology(N,tx,ix,x,y);
	for z=1:length(vS) % number of sources
		% Generate the packets
		fprintf('z = %d\n',z)
		BOwn = genData(P,p,upT,vS(z),nPkts);
		[engy(:,1),dly(1)] = dmac(N,T,upT,Tx,Ix,BOwn,0);
		% k==2 && z == 1
		try
			[engy(:,2),dly(2)] = orneMAC(N,T,upT,Tx,Ix,BOwn,P,x,y,0,k,z);
		catch
			[engy(:,2),dly(2)] = orneMAC(N,T,upT,Tx,Ix,BOwn,P,x,y,0,k,z);
		end
		for q = 1:2
			e(z,q) = e(z,q) + sum(engy(:,q)) / nTop;
			d(z,q) = d(z,q) + dly(q) / nTop;
		end
	end
end
toc
%%
figure(1)
plot(vS,e(:,1),vS,e(:,2))
title('Energy consumption')
ylabel('Energy (Joule)')
xlabel('Number of sources')
legend('dmac','RAGMAC')
figure(2)
plot(vS,d(:,1),vS,d(:,2))
title('Delay')
legend('dmac','RAGMAC')