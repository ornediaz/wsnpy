clear classes
clear global
clc, close all,format compact;

N = 4; % Number of nodes
global nL; % List of nodes
global t ; % current time
t = 0;
global mu ; % Slot time in DMAC
mu = 10;
global numTiers
global eventList


nL = cell(N,1);
interfLst = cell(N,1)
for k = 1:N
	nL{k,1} = snode(k);
end
nL{1}.tier = 2;
nL{2}.tier = 2;
nL{3}.tier = 2;
nL{4}.tier = 1;
interfLst{1} = 2;

numTiers = 2;

for k = 1:N
	nL{k} = setListening(nL{k});
end
[nL{1} event] = createPackets(nL{k},4);
[nL{2} event] = createPackets(nL{k},4);

eventList = struct('time',2.5,'node',1);
eventList(2) = struct('time',5,'node',2);
while ~isempty(eventList)
	[C,I] = min([eventList.time]);
	t = eventList(I).time;
	node = eventList(I).node;
	fprintf('t = %3.1f, node = %3.0f\n',t,node)
	eventList(I)=[];
	%processEvent(c{1})
end        



