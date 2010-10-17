% Runs txPhase.m in default topology
clear,clc,close all
rand('twister', 5487);
te = tree;
nS = 2;
nPkts = 8;
te = genData(te,nS,nPkts);

ac = linspace(0,1,5); % Aggregation coefficient
T = 20; %Number of averages
e = zeros(1,length(ac));
d = e;
for k = 1:T
	for z = 1:length(ac)
		clc
		[engy,dly] = txPhase(te,ac(z),false);
		e(z) = e(z) + sum(engy) / T;
		d(z) = d(z) + dly / T;
	end
end

plot(ac,d)
ylabel('Delay (second)')
xlabel('Aggregation coefficient \alpha')