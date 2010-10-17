function merge
clear,clc, close all,format compact%,rand('twister', 5489)
tic
set(0,'DefaultFigureWindowStyle','docked')
N=20;
BOwn = cell(1,N);
xSide = 200;
ySide = 20;
txRg = 60; % Radio transmission range
ixRg = 150; % Radio interference range
% Debugging topology %{{{
%N = 6;
%BOwn = cell(1,N); % Self-sensed packets. FIFO.
%interfRng = { 5 5 6 6 [1 2] [3 4] };
% Nodes within interference range
%tier = [3 3 2 2 1 0];
%trnRng = { [2 3 4] [1 3 4] [1 2 4 5] [1 2 3 5] [3 4 6] 5 };
%upTierID = { [3 4] [4 3] 5 5 6 [] };
% Next tier neighbors. 1st is default
%%}}}

[tier,upTierID,interfRng,trnRng,P,previous] = ...
	topology(N,txRg,ixRg,xSide,ySide);
figure
plt(P,previous,xSide,ySide)

for z = [5 8 9 4 10 12]  % Name of the node that originates the traffic
	for gg = 1:24; % Packet number
		% Append the packet in the output buffer
		BOwn{z} = [BOwn{z} newPkt(z,upTierID{z}(1),DataT,'DataK',gg,z)];
	end
end
engy = zeros(N,2);
dly = zeros(1,2);
[engy(:,1),dly(1)] = dmac(N,tier,upTierID,trnRng,interfRng,BOwn);
[engy(:,2),dly(2)] = orneMAC(N,tier,upTierID,trnRng,interfRng,BOwn,P,xSide,ySide);
fprintf('DMAC:: engy = %7.4f, delay = %7.4f\n',sum(engy(:,1)),dly(1));
fprintf('OMAC:: engy = %7.4f, delay = %7.4f\n',sum(engy(:),dly(2));
figure(3)
plot(1:N,[engy(:,1); engy(:,2)])
legend('DMAC','OMAC')
figure(4)

end
function [tier,upTierID,interfRng,trnRng,P,previous] = ...
	topology(N,txRg,ixRg,xSide,ySide) %{{{
interfRng = cell(1,N);
upTierID = cell(1,N);
trnRng = cell(1,N);

P=rand(N,2) .* repmat([xSide ySide],N,1);


INF=1000;
d=ones(1,N)*INF;
tier = Inf(1, N);
root = 1;
P(root,:) = [0 0];
d(root)=0;
tier(root) = 0;
previous=-1*ones(1,N);
% If element 'i' of temp is zero, it means it has not been processed yet
temp=zeros(1,N);
for kk=1:N %For all vertices
	[m u]=min(d+temp*2*INF);
	temp(u)=1;
	for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
		dist=norm(P(u,:)-P(jj,:)); % distance to 'jj' from 'u'
		if dist > txRg
			dist=INF;
		end
		alt=d(u)+dist;
		if (alt < d(jj))
			d(jj) = alt;
			previous(jj) = u;
		end
	end
end
% Plot the tree


% Compute each node's tier
for x=1:N
	p = x;
	n = 0;
	while p ~= -1 && p~=root
		n = n+1;
		p = previous(p);
	end
	if p ~= -1
		tier(x) = n;
	end
end
% if ~isempty(tier == Inf)
% 	error('Some nodes are in an infinite tier')
% end
for x = 1:N
	for y = [1:x-1 x+1:N]
		di = norm(P(x,:)-P(y,:));
		if di < txRg
			trnRng{x} = [trnRng{x} y];
			if tier(y) == tier(x) - 1
				upTierID{x} = [upTierID{x} y];
			end
		elseif di < ixRg
			interfRng{x} = [interfRng{x} y];
		end
	end
end

end %}}}
function [engy,dly] = dmac(N,tier,upTierID,trnRng,interfRng,BOwn) % {{{1
%% Init {{{
BIn = cell(1,N);
BOut = BOwn; % Packets to be sent. FIFO.
BOwn = BIn; %This is necessary to make moreToSend work
debugdmac = false;
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
	if debugdmac
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
			if tier(x) == 0 && debugdmac
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
end % }}}1 END OF DMAC
function [engy,dly] = ...
	orneMAC(N,tier,upTierID,trnRng,interfRng,BOwn,P,xSide,ySide) %{{{1
%{{{2

debugOrneMAC = false;
BIn = cell(1,N);
BOut = cell(1,N); % Packets to be sent. FIFO.
chld = cell(1,N);
engy = zeros(1,N); % Energy consumed
interfEndT = zeros(1,N); % End of interference
prnt = -ones(1,N);%Parent
prty = ones(1,N); % Priority relative to children. If 1, there are no
% siblings precede it
prntRdy = ones(1,N) < 0;% Parent acknowledged previous round?
radioT = zeros(1,N); % Time when current state started
radioSt = cell(1,N);for z=1:N,radioSt{z}='Slp';end
st = cell(1,N); for z =1:N,st{z}='SlpSt';end
upTierNoC = cell(1,N);% The numberOfChildren in the upper tiers


%% Create traffic and initial events {{{2
t = 0;
evntL= [];
% Schedule initial events for all nodes with children
for z = 1: N
	if tier(z)~= max(tier)
		evnt( nxtL4Ch(tier(z),max(tier),t) , z, 'stLk4Ch');
	elseif ~isempty(BOwn{z})
		evnt( nxtL4Pt(tier(z),max(tier),t), z, 'stLk4Pt');
	end
end
%% Event loop %{{{2
numEvents = 0;
while ~isempty(evntL)
	numEvents = numEvents + 1;
	if numEvents > 1e5
		break
	end
	[t,ind] = min([evntL.time]);
	if t > 30
		error('This simulation has taken too long')
	end
	e = evntL(ind);
	x = e.node;
	evntL(ind)=[];
	if debugOrneMAC
		%disp('****************')
		%disp('Current event')
		fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,e.type);
		%disp('Current event list')
		%dispEvntL
		%fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,e.type);
	end

	% 	if ~isempty(BOut{1}) && ~isempty(BOut{1}.Nr) && BOut{1}.Nr == 4
	% 		disp('Hi')
	% 	end
	%length(BOut{12}) == 1 && ~isempty(BOut{12}.Nr) && BOut{12}.Nr ==3 &&
	%prntRdy(12)
	switch e.type
		case 'endPktRx' %{{{
			p = BIn{x}(end);
			setRadio(x,'Idl');
			if interfEndT(x) < t - p.drtn
				% The packet was received without interference
				receivePacket(p,x)
			else
				% Unsuccessful reception due to interference
				BIn{x}(end) = [];
			end
			%}}}
		case 'endPktTx' %{{{
			p = BOut{x}(1);
			setRadio(x,'Idl')
			switch p.ty
				case {'RelReqK','DataK'} %{{{
					% Retransmit until ACK reception.
					evnt( t + backoffT(x), x, 'inPktTx');
					%}}}
				case 'AckK' %{{{
					% Remove the ACK from output buffer.
					BOut{x}(1) = [];
					% We have acknowledged a packet reception. Now (and not
					% b4) we process the received packet.
					
					%Discard duplicated packets
					if length([BIn{x}.frm]) > 1
						if sum(BIn{x}(end).frm == [BIn{x}(1:end-1).frm])
							BIn{x}(end) = [];
						end
					end

					%If all needed packets, aggregate and transmit
					% Check if I have all packets
					allPkts = true;
	
					for z = chld{x}
						if ~sum([BIn{x}.frm] == z)
							allPkts = false;
						end
					end
					if length(BIn{x}) == length(chld{x}) && ~allPkts
						error('Correct # of pkt, but unexpected ones')
					end
					% If I have all the packets, aggregate them
					if allPkts
						% Aggregate data
						% Create list of all packets to be aggregated
						packets = [BIn{x}];
						if ~isempty(BOwn{x})
							packets = [packets BOwn{x}(1)];
							BOwn{x}(1) = []; %Remove it
						end
						if tier(x) && ~isempty(BOut{x})
							error('BOut should be empty')
						end
						BOut{x} =[BOut{x} aggregate(packets,x,prnt(x))];
						% Set the TxSt state
						BIn{x} = [];
						if tier(x) % If not tier 0
							setState(x,'TxSt');
						else
							% If in tier 0, never transmit
							setState(x,'Wt4D');
							ackChld(x,p.Nr);
						end
						% Remove packet from input buffer
					end
					%}}}
				case 'RelOffK'
					% Remove RelOff pkt from output buffer.
					BOut{x}(1) = [];
				otherwise
					error('Unexpected packet type')
			end
			%}}}
		case 'inPktTx' % Initiate packet transmission {{{
			% A node enters this state when it has taken the
			% decision to transmit. I have introduced this event in
			% order to take into account the delay involved in
			% carrier sensing.
			setRadio(x,'Slp');
			% A node whose radio is in SlpSt will not cancel its
			evnt( t + CarrSensT, x, 'startPktTx' );
			%}}}
		case 'stLk4Ch' %{{{
			setState(x,'Lk4Ch');
			%}}}
		case 'stLk4Pt'%{{{
			if strcmp(st{x},'Lk4Ch') && ...
					isempty(BOwn{x}) && isempty(chld{x})
				% No data or children w data. Sleep rather than look 4
				% parent.
				setState(x,'SlpSt');
			elseif ~tier(x) % If we are in tier 0
				setState(x,'Wt4D')		
				figure(2)
				plt(P,prnt,xSide,ySide)
			else %We need to look for a parent
				% Tell my children their priority
				for z = 1:length(chld{x})
					prty(chld{x}(z)) = z;
				end
				setState(x,'Lk4Pt');
			end
			%}}}
		case 'stTxSt' % Start transmitting state{{{
			% I'm pkt src w/o children, this is my 1st tx
			BOut{x} = BOwn{x}(1);
			BOwn{x}(1) = [];
			% My parent is necessarily ready
			prntRdy(x) = true;
			setState(x,'TxSt');
			%}}}
		case 'stWt4D' %{{{
			% We only execute this after the setup phase.
			prntRdy(x) = true;
			setState(x,'Wt4D');
			%}}}
		case 'startPktTx'%{{{
			p = BOut{x}(1);
			setRadio(x,'Tx')
			evnt(t+p.drtn,p.frm,'endPktTx')
			%For nodes within interference range
			for z = [ interfRng{x} trnRng{x} ]
				interf(x,z,p)
			end
			% }}}
		otherwise
			%error('Unexpected event: %s',e.type)
	end
end %while
%% NESTED FUNCTIONS %{{{2
	function ackChld(x,Nr)
	for y = chld{x}
		prntRdy(y) = true;		
		% My children can initiate the next round.
		if length(BOut{y}) > 1
			error('length(BOut) should be smaller than 1')
			% Cancel retx of any packet whose Nr is smaller than Nr
		elseif length(BOut{y}) == 1 && strcmp(BOut{y}.ty,'DataK') ...
				&& BOut{y}.Nr==Nr
			% delete it
			BOut{y} = [];
			removeEvent(y,'inPktTx');
		end

		if ~moreToSendInSubTree(y,chld,BOut,BOwn)
			setState(y,'SlpSt')
		elseif isempty(chld{y}) && isempty(BOut{y})
			% No children but something in BOwn.
			% The node is a source that does not need to merge the data.
			BOut{y} = BOwn{y}(1);
			BOwn{y}(1) = [];
			setState(y,'TxSt');
		elseif ~isempty(BOut{y}) %It has a packet available for transmission
			% The node has children
			setState(y,'TxSt');
		elseif strcmp(st{y},'TxSt')
			%Its output buffer is empty and it needs to get data from children
			setState(y,'Wt4D')
			for q=chld{y}
				if ~prntRdy(q)
					prntRdy(q) = true;
					if strcmp(st{q},'TxSt')
						setState(q,'TxSt')
						% This way it will start transmitting if it was waiting for
						% its parent to be ready.
					end
				end
			end
		end
	end
	for w = find(prnt == find(tier == 0));
		if ~moreToSendInSubTree(w,chld,BOut,BOwn)
			dly = t;
			return
		end
	end
	end

	function t = backoffT(x) %{{{
	t = FxdBkffT + ContentionT* (1 + ( prty(x) - 1 ) * .2 ) * rand;
	end %}}}
	function evnt(time, x, type) %{{{
	for w = 1:length(evntL)
		if strcmp(evntL(w).type,type) && evntL(w).node == x
			error('Duplicate event')
		end
	end
	evntL = [evntL struct('time',time,'node',x,'type',type)];
	end %}}}
	function dispEvntL %{{{
	for kk = 1:length(evntL)
		fprintf('t = %7.4f, x = %3.0f, type = %s \n',...
			evntL(kk).time , evntL(kk).node,evntL(kk).type)
	end
	end %}}}
	function interf(x, y, p) %{{{
	% x = transmitter, y = interfered, p = packet
	% If within transmission range and idle, make them Rx.
	bo = false;
	if sum( y == trnRng{x} ) && strcmp(radioSt{y},'Idl') ...
			&& interfEndT(y) < t % (no interference)
		% Place the packet into the input buffer.
		BIn{y} = [BIn{y} p];
		setRadio( y ,'Rx');
		evnt( t + p.drtn, y, 'endPktRx' )
		bo = true; % backing off but interrupted
		% Otherwise interfere them
	elseif t + p.drtn > interfEndT(y)
		% If new interference finishes later than previous one,
		% readjust the end of interference time
		interfEndT(y) = t + p.drtn;
	end
	% Handle nodes backing off
	eventEnd = max( interfEndT(y), t + p.drtn );
	if  strcmp(st{y},'Lk4Pt') && eventEnd + NdStpT > endTierT(t)
		% No more time to look for a parent.
		error('I did not find a parent')
	elseif strcmp(st{y},'Lk4Pt') && (bo || strcmp(radioSt{y},'Idl') )...
			||  strcmp(st{y},'TxSt') && prntRdy(y) && (bo || strcmp(radioSt{y},'Idl') )
		% && has priority over ||
		% The check of the radio is important. If it is not Idl, it may be because
		% it executed an inPktTx event but not yet an startPktTx event

		% Reschedule the transmission
		found = [];
		for jj=1:length(evntL)
			if evntL(jj).node == y && strcmp(evntL(jj).type,'inPktTx')
				found = [found jj];
			end
		end
		if length(found) > 1
			error('I found too many elements to reschedule')
		elseif isempty(found)
			error('I found no event to reschedule')
		end
		evntL(found).time = eventEnd + backoffT(y);
	end
	end %}}}
	function receivePacket(p,x) %{{{
	switch st{x}
		case 'Lk4Ch' %{{{
			% Remove packet from input buffer
			BIn{x}(end) = [];
			if p.dst == x
				switch p.ty
					case 'RelReqK'

						% If not already a child, add to child list.
						if ~sum(chld{x} == p.frm)
							chld{x}(end+1) = p.frm;
						end

						% Reply with an RelOffK packet
						BOut{x} = [newPkt(x,p.frm, RelOffT ,'RelOffK',p.Nr,p.src) BOut{x}];
						setRadio(x,'Slp') % This way a node cannot start receiving
						evnt( t + B4AckT , x,'inPktTx')
					otherwise
						% If packet addressed to itself, it can only be of that type
						error('I should not be receiving this packet')
				end
			end
			%}}}
		case 'Lk4Pt' %{{{
			% Remove packet from input buffer
			BIn{x}(end) = [];
			switch p.ty
				case 'RelOffK'
					if p.dst == x
						if debugOrneMAC
							disp('==========================')
							fprintf('Node %d found % d as parent\n',x,p.frm)
							disp('==========================')
						end
						% I have obtained a parent.
						prnt(x) = p.frm;
						for kk = 1:length(BOwn{x})
							BOwn{x}(kk).dst = prnt(x);
						end
						% Cancel retransmission
						removeEvent(x,'inPktTx')
						BOut{x} = [];

						% Schedule next event
						setState(x,'SlpSt');
						if isempty(chld{x})
							% No need to listen for data.
							% Tx when setup phase of next 3 tiers over.
							evnt(endTierT(t) + nt * TierStpT,x,'stTxSt')
						else
							evnt(endTierT(t) + (nt-1) * TierStpT,x,'stWt4D')
						end
					else
						% Update next tier's number of children
						ii = find ( upTierID{x} == p.frm );
						upTierNoC{x}(ii) = upTierNoC{x}(ii) + 1;
						% Set relay with more children as RelReq destination
						[C ii] = max( upTierNoC{x} );
						BOut{x}(1).dst = upTierID{x}(ii);
					end
				case 'RelReqK'
					setRadio(x,'Idl');
				otherwise
					error('I should not be receiving this packet')
			end
			%}}}
		case 'TxSt' %{{{
			BIn{x}(end) = []; % Delete pkt from input buffer.
			if p.dst == x
				switch p.ty
					case 'AckK'
						% PROCESS MYSELF
						%Cannot initiate the next round. Must wait 4 my parent to rx Ack
						%from its parent.
						if length(BOut{x}) ~= 1
							error('length(BOut) should be 1')
						elseif ~strcmp(BOut{x}.ty,'DataK')
							error('There should be a DataK pkt in BOut')
						end
						% execute the code below only if there is a packet with
						% number <= than current
						if BOut{x}.Nr == p.Nr
							BOut{x}= [];
							removeEvent(x,'inPktTx');
							prntRdy(x) = false;
							if ~moreToSendInSubTree(x,chld,BOut,BOwn)
								setState(x,'SlpSt')
							elseif isempty(chld{x}) && isempty(BOut{x})
								% No children but something in BOwn.
								% The node is a source that does not need to merge the data.
								BOut{x} = BOwn{x}(1);
								BOwn{x}(1) = [];
								setState(x,'TxSt');
							elseif ~isempty(BOut{x}) %It has a packet available for transmission
								% The node has children
								setState(x,'TxSt');
							else %Its output buffer is empty and it needs to get data from children
								setState(x,'Wt4D')
							end
						end
						% PROCESS MY CHILDREN
						ackChld(x,p.Nr);
					case 'DataK'
						% A child has retransmitted a packet unnecessarily.
						sendAck(x,p);
					otherwise
						error('Unexpected packet type')
				end
			end
			%}}}
		case 'Wt4D' %{{{

			if p.dst == x &&strcmp(p.ty,'DataK')
				sendAck(x,p)
			else
				BIn{x}(end) = []; % Delete pkt from input buffer.
			end
			%}}}
		case 'SlpSt'
			if ~isempty(BOut{x}) || ~isempty(BOwn{x})
				error('Why is it sleeping?')
			end
		otherwise
			error('Unexpected state')
	end %switch
	end %}}}
	function removeEvent(x,type) %{{{
	found = removeEventWithoutError(x,type);
	if length(found) >1
		error('I found too many elements to remove')
	elseif isempty(found)
		error('I found no events to remove')
	end
	end %}}}
	function found = removeEventWithoutError(x,type) %{{{
	found = [];
	for kk=1:length(evntL)
		if evntL(kk).node == x && strcmp(evntL(kk).type,type)
			found = [found kk];
		end
	end
	evntL(found) = [];
	end %}}}
	function sendAck(x,p) %{{{
	BOut{x} = [newPkt(x,p.frm, AckT,'AckK', p.Nr,p.src) BOut{x}];
	setRadio(x,'Slp'); % This way a node cannot start receiving
	evnt( t + B4AckT , x,'inPktTx');
	end %}}}
	function setRadio(x,newState) % {{{
	% Compute the energy consumed
	switch newState
		case {'Idl' 'Rx' 'Tx' 'Slp'}
		otherwise
			error('Incorrect radio type')
	end
	switch radioSt{x}
		case 'Rx'
			pow = PowerReceive;
		case 'Tx'
			pow = PowerTransmit;
		case 'Idl'
			pow = PowerIdle;
		case 'Slp'
			pow = PowerSleep;
		otherwise
			error('I did not find the consumption for this state')
	end
	engy(x) = engy(x) + (t - radioT(x)) * pow;

	% Set new state
	radioSt{x} = newState;

	% Record the time when the state started to compute the energy
	% consumption.
	radioT(x) = t;
	end %}}}
	function setState(x,newState) %{{{
	if strcmp(newState,'Wt4D')
		if prnt(x) > 0 && isempty(chld{x})
			error('Leaves should not wait for data')
		end
	end
	switch newState
		case 'Lk4Ch' % Looking4Children {{{
			setRadio(x,'Idl');
			chld{x} = [];
			% Schedule end of state: Lk4Pt or SlpSt
			evnt( t + TierStpT, x , 'stLk4Pt' ); %}}}
		case 'Lk4Pt' % Looking4Parent{{{
			setRadio(x,'Idl');
			% Clear parent
			prnt(x) = -1;
			% Clear the count of next-tier neighbor's children.
			upTierNoC{x} = zeros(size(upTierID{x}));
			% Create the new relay request packet
			BOut{x} = newPkt(x,upTierID{x}(1),RelReqT,'RelReqK',[],[]);
			% Schedule the transmission
			evnt( t + backoffT(x), x, 'inPktTx' );
			% endLk4Pt not needed, bc tx considers nodes unable 2 find parent.
			%}}}
		case 'SlpSt' % Sleep State{{{
			setRadio(x,'Slp');
			%}}}
		case 'TxSt' %{{{
			% In this state, the node has a packet ready in the output buffer.
			setRadio(x,'Idl');
			if prntRdy(x)
				evnt( max(t,interfEndT(x)) + backoffT(x), x, 'inPktTx')
			end
			%}}}
		case 'Wt4D' % Wg4Data{{{
			setRadio(x,'Idl');

			% 	if ~strcmp(st{x},'Wt4D')
			% 		%It was not already in this state
			% 		BIn{x}= [];
			% 	end
			%}}}
		otherwise %{{{
			error('Unexpected state name');
			%}}}
	end %switch
	st{x} = newState;

	end

%}}}
toc,
end % END OF orneMAC
% Independent functions {{{1
function p = aggregate(packets,x,y) %{{{
if var([packets.Nr]) > 1e-3
	error('All packets should be from the same round')
end
drtn = DataT * log2(1+length([packets.src]));
p = newPkt(x,y,drtn,'DataK',packets(1).Nr,[packets.src]);
end %}}}
function tc = endTierT(t) %{{{
% Time when the current tier setup phase ends
t0 = floor(t/CompletePeriod) * CompletePeriod;
tc = t0 + ceil( (t-t0) / TierStpT )*TierStpT;
end %}}}
function b = moreToSendInSubTree(x,chld,BOut,BOwn) %{{{
b = ~isempty([BOut{x} BOwn{x}]);
for z = chld{x}
	b = b + moreToSendInSubTree(z,chld,BOut,BOwn);
end
end %}}}
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
function h = nxtL4Pt(tier,TreeDepth,t)%{{{
t0 = floor(t/CompletePeriod) * CompletePeriod;
h = t0 + (TreeDepth - tier) * TierStpT;
end %}}}
function h = nxtL4Ch(tier,TreeDepth,t) %{{{
t0 = floor(t/CompletePeriod) * CompletePeriod;
h = t0 + (TreeDepth - tier - 1) * TierStpT;
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
function h = newPkt(frm,dst,drtn,ty,Nr,src) %{{{
switch ty
	case {'RelReqK','RelOffK','DataK','AckK'}
		h = struct(...
			'frm',frm,...
			'dst',dst,... % destination
			'drtn',drtn,... %duration
			'ty',ty,...
			'Nr', Nr,...
			'src',src);
	otherwise
		error('Unexpected packet type')
end %switch
end %}}}

%% CONSTANT DEFINITION {{{
% In DMAC:
% mu = FxdBkffT + ContentionT + DataT + B4TxAckT + AckT;
% We need FxdBckff > B4AckT
%Set timing parameters(FxdBkffT + ContentionT + DataT + B4AckT + AckT)

function h = AckT, h = 1e-3;end % Acknowledgement transmission time
function h = B4AckT, h = 3e-4; end
function h = CarrSensT, h = 0.1*ContentionT;
% Time 2 detect an ongoing tx. Two nodes that decide to transmit during
% this time will generate a collision if the recipient is within both
% nodes' range.
end
function h = CompletePeriod; h = 1;
% Time between the beginning of 2 natural listen periods in DMAC and
% OrneMAC.
end
function h = ContentionT; h= 1e-3; end % Contention window time
function h = DataT, h = 8e-3; end % Data time
function h = FxdBkffT, h = 4e-4;
if h <=B4AckT
	error('The following does not hold: FxdBckffT > B4AckT')
end
end
function h = mu, h = 11e-3;end %Period time
function h = NdStpT
% Setup time per tier used in OrneMAC
h = FxdBkffT + 2* ContentionT + RelReqT + B4AckT + RelOffT;
end
function h = nt, h = 4;
%In OrneMAC, number of TierStpT to wait after finding a parent before
%starting listening for children data. It should be at least three when
%intef range doubles transmission range.
end
function h = PowerTransmit, h = 0.66; end
function h = PowerReceive, h = 0.395; end
function h = PowerIdle, h = 0.35; end
function h = PowerSleep, h = 0; end
function h = RelOffT, h = AckT; end % Time it takes to send a RelOffK
function h = RelReqT, h = AckT;end % Time it takes to send a RelReqK.
% A node will only transmit one packet every 5*mu
function h = TierStpT, h =  10 * NdStpT;
%NdStpT * nContend;
end
function h = TransmitPeriod, h = 5 * mu;
%Time between the beginning of two consecutive listening in the
%increased duty cycle in DMAC
end


% vim:foldmethod=marker:ts=4:sw=4:tw=76
