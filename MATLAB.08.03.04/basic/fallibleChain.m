%% Delay and queue length in a fallible chain of nodes
function fallibleChain
clc,close all
% set(0,'defaultAxesFontSize',8)
% set(0,'defaultTextFontName','Calibri')
set(0,'defaultFigureWindowStyle','docked')
rand('state',0)
tic

boolPrint=0;

delete(sprintf('%sFig*',mfilename))
verbose = 0; % Print the results of each simulation?

N=12; %Number of nodes in the chain
p=0.5:.1:1; %Probability of successs
numPackets=1000;
buffQueue=zeros(10*numPackets,length(p));
delay=zeros(numPackets,length(p));
efi=zeros(1,length(p));
numIt = zeros(length(p),1);
rand('state',0)
for k=1:length(p)
	%[queueSize,efi(k),delay(:,k)] = simStd(N,p(k),numPackets,verbose);
	[delay(:,k),numIt(k),efi(k),queueSize] = simEng(N,p(k),numPackets,verbose);
	buffQueue(1:length(queueSize),k)=queueSize;
	leg(k,:)=sprintf('p = %1.2f',p(k));
end
% fprintf('\nEfficiency = %f\n',efi)
toc

figure,clf
plot(p,efi)
title('Increased delivery time due to errors')
ylabel('(delivery time)/(delivery time without errors)')
xlabel('Link success probability')
myPrint(boolPrint)

figure,clf
plot(buffQueue(1:max(numIt),:))
title('Maximum queue length in relay nodes')
ylabel('maximum queue length')
xlabel('Iteration number number')
legend(leg)
myPrint(boolPrint)

figure,clf
plot(delay)
legend(leg)
xlabel('Packet number')
ylabel('Extra delay in timeslots')
title('Delay due to retransmissions or queueing')
myPrint(boolPrint)


%%
% Simulate the number of retransmission necessary in a pipeline in which at
% each step one tries until one succeeeds. This random variable is a
% negative binomial distribution. I compare the simulation with the theory.
clear,clc
N = 3; % Number of hops
p = 0.5;
numSim = 1e3;
histo=zeros(1,100);
for nSim=1:numSim
    countHop=zeros(1,N);
    for k=1:N
        while rand > p
            countHop(k) = countHop(k)+1;
        end
    end
    countGlobal=sum(countHop);
    histo(countGlobal+1)=histo(countGlobal+1)+1;
end
histo=histo/numSim;
n=1:10;
figure,clf
subplot(1,2,1)
stem(n-1,histo(n))
y=nbinpdf(n-1,N,p);
subplot(1,2,2)
stem(n-1,y)
% hold on
% % E[Delay] if there were no queuing delays
% meanDelWithoutCol=N*(1-p)./p;
% plot(meanDelWithoutCol,'*')

%%
function [delay,numIt,efi,queueSize] = simEng(N,p,numPackets,verbose)

c = cell(1,N);
for k=2:N;
	c{k} = [];
end
c{1} = 1:numPackets;

numIt=0;
finished = 0;
timeArrival = zeros(1,numPackets);

maxNumIt = 100 * numPackets;
queueSize = zeros( 1 , maxNumIt );
while numIt < maxNumIt && ~finished
	numIt=numIt+1;
	for numNode = mod( numIt - 1, 3 ) + 1 : 3 :N
		if ~ isempty( c{numNode} ) % The node "numNode" has a packet in its buffer
			if rand < p % The transmission succeeded
				% Extract first packet from the queue (buffer,index)
				pkt = c{numNode}(1);
				c{numNode}(1) = [];
				if numNode == N % The packet reached the sink
					timeArrival(pkt) = numIt;
					if pkt == numPackets
						finished = 1;
					end
				else % The packet did not reach the sink
					c{numNode + 1}(end+1) = pkt;
				end
			end	% If the transmission failed, do nothing
		end
	end % All the nodes that could transmit have already done so
	if verbose
		fprintf('\nIteration %4d: ',numIt);
	end
	% Write the queue length in each buffer
	longestQueueSize = 0;
	for k=1:N
		qs = length( c{k} );
		if verbose
			fprintf( '%2d-', qs)
		end
		if qs > longestQueueSize && k~= 1
			longestQueueSize = qs;
		end
	end
	if verbose
		for k=1:N
			fprintf( '\nNode %d queue: ', k )
			fprintf( '%2d-', c{k} )
		end
		fprintf('\n')
	end
	queueSize(numIt) = longestQueueSize;
end
fprintf('\n Number of slots: %d\n', numIt)
if ~finished
	disp('Not all packets were shipped')
end
delay=timeArrival-N-(0:numPackets-1)*3;
%  stem( timeArrival )
% Efficiency, defined as the quotient between the number of slots that the
% system required and the minimum number of slots that we could achieve if
%  all transmissions succeeded.
efi = numIt/(N+(numPackets-1)*3);


function myPrint(bool)
if bool
stack=dbstack;%
callerFileName=stack(end).name;
fontN='Calibri';
fontS=8;
prop=struct('FontName','Calibri','FontSize',8);
set(gca,prop)
set(get(gca,'XLabel'),prop)
set(get(gca,'YLabel'),prop)
set(get(gca,'Title' ),prop)
set(gcf,'PaperUnits','centimeters', 'PaperPosition',[0 0 15 10])
print('-dpng',sprintf('%sFig%02d',callerFileName,gcf))
end