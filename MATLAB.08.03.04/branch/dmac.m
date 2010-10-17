function [engy,dly] = dmac(N,tier,upTierID,trnRng,interfRng,BOwn,verbose) 
%% Init {{{
BIn = cell(1,N);
BOut = BOwn; % Packets to be sent. FIFO.
BOwn = BIn; %This is necessary to make moreToSend work
engy = zeros(1,N); % Energy consumed
interfEndT = zeros(1,N); % Time when interference ends
oneMore = zeros(1,N);
parent = zeros(1,N);
chld = cell(1,N);
for jj=1:N
	if isempty(upTierID{jj})
		parent(jj) = -1;
	else
		parent(jj) = upTierID{jj}(1);
		chld{parent(jj)} = [ chld{parent(jj)} jj ];
	end
end
radioT = zeros(1,N); % Time when current state started
st = cell(1,N); for z =1:N,st{z}='SleepState';end

%% Schedule the initial listening event for all nodes with children
eventList= [];
t = 0;
for jj = 1: N
	if ~isempty(chld{jj})
		evnt(firstRxTime(tier(jj),max(tier)), jj, 'stWg4D');
	elseif ~isempty(BOut{jj})
		evnt(nextNaturalTxTime(t,tier(jj),max(tier)), jj, 'stCont');
	end
end


%% Event process loop
numEvents = 0;
%}}}
while ~isempty(eventList)
	numEvents = numEvents + 1;
	if numEvents > 1e6, break, end
	[t,ind] = min([eventList.time]);
	e = eventList(ind);
	x = e.node;
	eventList(ind)=[];
	if verbose
		fprintf('t = %7.4f, node = %3.0f, type = %s \n',t,x,e.type);
	end
	switch e.type
		case 'endAckRx' %{{{
			% Remove the last received packet
			p = BIn{x}(end);
			BIn{x}(end) = [];

			if interfEndT(x) < t - AckT  && p.frm == parent(x)
				if ~strcmp(p.ty,'AckK')
					error('I was expecting an AckK packet')
				end
				% The packet was successfully received. Remove the first
				% packet from the output buffer
				BOut{x}(1) = [];
			end
			%enter into Idle state immediately
			setState(x,'IdleState')
			% Do not schedule any event. The next event for this node will be
			% 'endCont'.  I want to leave the node in a high energy
			% consumption state. }}}
		case 'endAckTx', setState(x,'IdleState');
		case 'endCont' %{{{
			% All nodes that contend, succcessful or not, reach this point.
			% We schedule the next listening event
			setState(x,'SleepState')
			if (oneMore(x) > 0)
				%Schedule the next increased listening.
				evnt(nextIncreasedRxTime(t,tier(x),max(tier)),x,'stWg4D');
			elseif ~isempty(BOut{x})
				%there are packets in the output buffer
				%Schedule the next increased transmitting
				evnt(nextIncreasedTxTime(t,tier(x),max(tier)),x,'stCont');
			elseif ~isempty(chld(x))
				% Schedule the next natural listen
				evnt(nextNaturalRxTime(t,tier(x),max(tier)),x,'stWg4D');
			end
			%}}}
		case 'endDataRx' %{{{
			setState(x,'IdleState');
			%Remove last node to enter the vector
			p = BIn{x}(end);
			% The packet will be removed except when the reception was correct
			% and the node has no parent.
			if interfEndT(x) > t - DataT
				% The reception was unsuccessful (the interference finished
				% after the reception started
				BIn{x}(end) = [];
			elseif	p.dst == x  % I am the recipient
				if ~strcmp(p.ty,'DataK')
					error('I should be receiving a DataK packet')
				end
				% The node will acknowledge the reception.
				ackPkt = newPkt(x,p.frm,AckT,'AckK',p.Nr,p.src);
				evnt(t + B4AckT, x,'startAckTx');
				% The Ack will be transmitted next.
				BOut{x} = [ackPkt BOut{x}];

				if parent(x) > 0
					% The node has a parent, therefore it will relay the packet
					% to it.
					BIn{x}(end) = [];
					relayPkt = newPkt(x,parent(x),AckT,'DataK',p.Nr,p.src);
					% The packet will be transmitted last
					BOut{x} = [BOut{x} relayPkt];
				end
			end
			%}}}
		case 'endWg4D' %{{{
			setState(x,'SleepState')
			if ~isempty(BOut{x})
				%Execute 'stCont' right now
				evnt(t,x,'stCont')
			elseif oneMore(x) > 0
				evnt(nextIncreasedRxTime(t,tier(x),max(tier)),x,'stWg4D');
			elseif ~tier(x)
				dly = t;
				return
			else
				evnt(nextNaturalRxTime(t,tier(x),max(tier)),x,'stWg4D');
				%Schedule 'stWg4D' @ natural listening time
			end
			if tier(x) == 0 && verbose
				disp('=================================')
				disp('End of cycle')
				disp('=================================')
			end
			%}}}
		case 'stWg4D' %{{{
			setState(x,'Waiting4DataState');
			if moreToSendInSubTree(x,chld,BOut,BOwn)
				oneMore(x) = 1;
			else
				%Decrement oneMore
				oneMore(x) = oneMore(x) -1;
			end
			evnt(t + mu,x,'endWg4D')
			%This event will never be removed.  When packet arrives: enter
			% receiving Data State and stay in it for DataT.

			%}}}
		case 'startAckTx' %{{{
			setState(x,'TransmittingAckState');
			% Transmit the Ack packet from the front of the output buffer and
			% remove it.
			transmitPacket(BOut{x}(1));
			BOut{x}(1) = [];
			evnt(t+ AckT,x,'endAckTx')
			%}}}
		case 'stCont' %{{{
			setState(x,'Bckff');
			evnt( t + mu * 0.999,x,'endCont')
			evnt( t + FxdBkffT + ContentionT * rand,x,'startDataTx' );
			% If data arrives while contending
			% Enter idle state and stay that way until the 'endCont' event, which
			% is going to be executed even if it transmits successfully.
			%}}}
		case 'endDataTx' %{{{
			setState(x,'Wg4Ack')
			% Do not schedule end time. At the worst case,
			%the 'endCont' will execute.
			% If a packet arrives while 'Wg4Ack'
			% Enter into 'ReceivingAckState' and schedule an 'endAckRx' event
			% AckT later.
			%}}}
		case 'startDataTx' %{{{
			setState(x,'TransmittingDataState');
			transmitPacket(BOut{x}(1))
			evnt(t + DataT, x, 'endDataTx');
			%}}}
		otherwise, error('Unrecognized event type')
	end
end
%% NESTED FUNCTIONS
	function setState(nodeIndex,newState) %{{{
	% Compute the energy consumed
	switch st{nodeIndex}
		case {'Bckff', 'IdleState','Waiting4DataState','Wg4Ack'}
			powerConsumption = PowerIdle;
		case { 'ReceivingDataState', 'ReceivingAckState'}
			powerConsumption = PowerReceive;
		case {'TransmittingDataState', 'TransmittingAckState'}
			powerConsumption = PowerTransmit;
		case {'SleepState'}
			powerConsumption = PowerSleep;
		otherwise
			error('I did not find the consumption for this state')
	end
	durationOfLastState = t - radioT(nodeIndex);
	engy(nodeIndex) = engy(nodeIndex) + durationOfLastState * powerConsumption;

	% Set new state
	st{nodeIndex} = newState;
	% Record the time when the state started to compute the energy
	% consumption.
	radioT(nodeIndex) = t;
	end %}}}
	function evnt(time, node, type) %{{{
	eventList = [eventList struct('time',time,'node',node,'type',type)];
	end %}}}
	function removeEvent(node,type) %{{{
	found = [];
	for kk=1:length(eventList)
		if eventList(kk).node == node && strcmp(eventList(kk).type,type)
			found = [found kk];
		end
	end
	if length(found) >1
		error('I found too many elements to remove')
	elseif isempty(found)
		error('I found no events to remove')
	end
	eventList(found) = [];
	end %}}}
	function transmitPacket(p) %{{{
	% This transmitting function is independent of the data packet. A
	% node, after concluding its reception, must check that it received
	% the right kind of packet.
	for kk = [interfRng{p.frm} trnRng{p.frm}] % for all interfered nodes
		switch st{kk}
			case 'Waiting4DataState'
				setState(kk,'ReceivingDataState')
				%Schedule 'endDataRx' event DataT later
				evnt(t + DataT*1.001, kk, 'endDataRx');
				% Place the node into the input buffer
				BIn{kk} = [BIn{kk} p];
			case 'Wg4Ack'
				% Set 'ReceivingAckState'
				setState(kk,'ReceivingAckState');
				%Schedule 'endAckRx' event AckT later
				evnt( t + AckT *1.001, kk, 'endAckRx' );
				BIn{kk} = [BIn{kk} p];
			case 'Bckff'
				% remove 'startDataTx' event
				removeEvent(kk,'startDataTx');
				setState(kk,'IdleState');
				% Do not schedule any event, as the node will stay in high energy
				% state until the the 'endCont' event
			otherwise
				interfEndT(x) = t + p.drtn;	% WAITING FOR TRANSMISSION STATES
		end
	end

	end %}}}
end 
function h = firstPossibleTxTime(tier,TreeDepth) %{{{
h = (TreeDepth - tier) * mu;
end %}}}
function h = firstRxTime(tier,TreeDepth) %{{{
h = (TreeDepth - tier - 1) * mu;
end %}}}
function h = nextNaturalTxTime( t, tier,TreeDepth ) %{{{
h = nextNaturalTime( t, firstPossibleTxTime(tier,TreeDepth) );
end %}}}
function h = nextNaturalRxTime(t,tier,TreeDepth) %{{{
h = nextNaturalTime( t, firstRxTime(tier,TreeDepth) );
end %}}}
function h = nextNaturalTime( t, firstNaturalTime) %{{{
h = firstNaturalTime + ceil( (t - firstNaturalTime) / CompletePeriod) * CompletePeriod;
end %}}}
function h = nextIncreasedTxTime(t,tier,TreeDepth) %{{{
h = nextIncreasedTime(t,nextNaturalTxTime(t,tier,TreeDepth));
end %}}}
function h = nextIncreasedRxTime(t,tier,TreeDepth) %{{{
h = nextIncreasedTime(t,nextNaturalRxTime(t,tier,TreeDepth));
end %}}}
function h = nextIncreasedTime(t,nextNaturalTime) %{{{
startIncreased = nextNaturalTime - CompletePeriod;
h = startIncreased + ceil( (t - startIncreased) / TransmitPeriod) * TransmitPeriod;
end %}}}
