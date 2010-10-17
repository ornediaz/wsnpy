function [engy,dly] = dmac(te,vrb)
% DMAC. It only has transmission phase.
%
% G. Lu, B. Krishnamachari, and C. S. Raghavendra, "An adaptive
% energy-efficient and low-latency MAC for data gathering in
% wireless sensor networks," in Proceedings of the IEEE 18th
% International Parallel and Distributed Processing Symposium (IPDPS
% '04) 2004, p. 224.
%
% It schedules the next listening or transmitting time
% appropriately. After a node has received a packet, it will listen
% in the next available slot, as opposed to having a full sleep
% period. If in this new slot it does not listen to anything, it
% will go to sleep for a long time.
%
% This program simulates the transmission of ACK messages. It
% does not consider the transmission and reception of More To Send
% (MTS) packets, which propagate towards the sink to indicate
% whether there are nodes down the tree with data. However, this
% program uses the information that a successful transfer of MTS
% packets would cause. The radio transceivers remain idle for the
% full mu time, to account the energy consumed in this process.
%
% The architecture considers many different states. For example,
% there is a different state for transmitting a DATA packet than for
% transmitting an ACK packet.
%
% Inputs:
% * te.N (double 1x1): number of nodes
% * te.tr (double 1xN): tier to which each node belongs
% * te.ch (cell 1xN): children of each node
% * tx (cell 1xN): list of nodes within tx range of each node
% * ix (cell 1xN): list of nodes within interference range
% * ot (cell 1xN): buffer with each nodes' packets
% * vrb (logical 1x1): print debug information
%
% Outputs:
% * engy (double 1xN): energy consumed by each node
% * dly (double 1x1): simulated time
in = cell(1,te.N);
ixT = zeros(1,te.N); % Time when interference ends
oneMore = zeros(1,te.N);
ot = cell(1,te.N);

r = radioStateManager(te.N);
s = cell(1,te.N);
for z =1:te.N
	s{z}='SleepState';
end

e = eventList(1e5);
t = 0;

for z = te.src(:)'
	for gg = 1:te.nPkts; % Packet number
		% Append the packet in the output buffer
		ot{z} = [ot{z} nPkt(z,te.pt(z),DataT,'DataK',gg,z)];
	end
end

%% Schedule the initial events
for jj = 1:te.N
	if ~isempty(te.ch{jj}) %Nodes with children
		e = ev( e,firstRxTime(te.tr(jj),max(te.tr)), jj, 'stWg4D' );
	elseif sum(te.src == jj) %Nodes with data
		e = ev( e, nxtNatTxT(t,te.tr(jj),max(te.tr)),jj, 'stCont');
	end
end
while ~isempty(e)
	[e,t,x,ty] = get(e);
	if vrb
		%disp('****************')
		%disp('Current event')
		fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,ty);
		%display(e)
	end
	if strcmp(ty,'endAckRx')
		% Remove the last received packet
		p = in{x}(end);
		in{x}(end) = [];

		if ixT(x) < t - AckT  && p.frm == te.pt(x)
			if ~strcmp(p.ty,'AckK')
				error('I was expecting an AckK packet')
			end
			% The packet was successfully received. Remove the first
			% packet from the output buffer
			ot{x}(1) = [];
		end
		%enter into Idle state immediately
		[r,s] = set(r,s,x,'IdleState',t);
		% Do not schedule any event. The next event for this node will
		% be 'endCont'.  I want to leave the node in a high energy
		% consumption state. }}}
	elseif strcmp(ty,'endAckTx')
		[r,s] = set(r,s,x,'IdleState',t);
	elseif strcmp(ty,'endCont')
		% All nodes that contend, succcessful or not, reach this
		% point. We schedule the next listening event.
		[r,s] = set(r,s,x,'SleepState',t);
		if (oneMore(x) > 0)
			%Schedule the next increased listening.
			e = ev(e,nxtIncRxT(t,te.tr(x),max(te.tr)),x,'stWg4D');
		elseif ~isempty(ot{x})
			%there are packets in the output buffer
			%Schedule the next increased transmitting
			e = ev(e,nxtIncTxT(t,te.tr(x),max(te.tr)),x,'stCont');
		elseif ~isempty(te.ch(x))
			% Schedule the next natural listen
			e = ev( e,nxtNatRxT(t,te.tr(x),max(te.tr)),x,'stWg4D');
		end
	elseif strcmp(ty,'endDataRx')
		[r,s] = set(r,s,x,'IdleState',t);
		%Remove last node to enter the vector
		p = in{x}(end);
		in{x}(end) = [];
		if ixT(x) > t - DataT || p.dst ~= x
			continue
		elseif ~strcmp(p.ty,'DataK')
			error('I should be receiving a DataK packet')
		end
		% The node will acknowledge the reception.
		e = ev(e,t + B4AckT, x,'startAckTx');
		ot{x} = [...
			nPkt(x,p.frm,AckT,'AckK' ,p.Nr,p.src) ...
			ot{x}...
			nPkt(x,te.pt(x),AckT,'DataK',p.Nr,p.src)];
	elseif strcmp(ty,'endWg4D')
		if te.tr(x) == 0 && vrb
			disp('=================================')
			disp('End of cycle')
			disp('=================================')
		end
		[r,s] = set(r,s,x,'SleepState',t);
		if ~isempty(ot{x}) && te.tr(x)
			% I have something to transmit and I am not the sink.
			%Execute 'stCont' right now to transmit pkt.
			e = ev(e,t,x,'stCont');
		elseif te.tr(x) && oneMore(x) || ~te.tr(x) && sum(oneMore(te.ch{x}))
			% I am not the sink  and I suspect my children have some
			% data or I am the sink and my children may have something
			% to send.
			e = ev(e,nxtIncRxT(t,te.tr(x),max(te.tr)),x,'stWg4D');
		elseif ~te.tr(x)
			% I am the sink and there is no need listen again until a
			% long sleep period has elapsed. This is the only way this
			% file should return.
			dly = t;
			engy = getE(r);
			return
		else
			e = ev(e,nxtNatRxT(t,te.tr(x),max(te.tr)),x,'stWg4D');
			%Schedule 'stWg4D' @ natural listening time
		end
	elseif strcmp(ty,'stWg4D')
		[r,s] = set(r,s,x,'Waiting4DataState',t);
		if moreToSendInSubTree(x,te,ot)
			oneMore(x) = true;
		else
			%Decrement oneMore
			oneMore(x) = false;
		end
		e = ev(e,t + mu,x,'endWg4D');
		%This event will never be removed.  When packet arrives: enter
		% receiving Data State and stay in it for DataT.
	elseif strcmp(ty,'startAckTx')
		[r,s] = set(r,s,x,'TransmittingAckState',t);
		% Transmit the Ack packet from the front of the output buffer
		% and remove it.
		[e,in,r,s,ixT] = txPkt(ot{x}(1),e,in,r,s,ixT,te,t);
		ot{x}(1) = [];
		e = ev(e,t+ AckT,x,'endAckTx');
	elseif strcmp(ty,'stCont')
		[r,s] = set(r,s,x,'Bckff',t);
		e = ev(e,t + mu * 0.99,x,'endCont');
		e = ev(e,t + FxdBkffT + ContentionT * rand,x,'startDataTx' );
		% If data arrives while contending, enter idle state and stay
		% that way until the 'endCont' event, which is going to be
		% executed even if it transmits successfully.
	elseif strcmp(ty,'endDataTx')
		[r,s] = set(r,s,x,'Wg4Ack',t);
		% Do not schedule end time. At the worst case,
		%the 'endCont' will execute.
		% If a packet arrives while 'Wg4Ack' Enter into
		% 'ReceivingAckState' and schedule an 'endAckRx' event AckT
		% later.
	elseif strcmp(ty,'startDataTx')
		[r,s] = set(r,s,x,'TransmittingDataState',t);
		[e,in,r,s,ixT] = txPkt(ot{x}(1),e,in,r,s,ixT,te,t);
		e = ev(e,t + DataT, x, 'endDataTx');
		%}}}
	else
		error('Unrecognized event type')
	end
end


function h = firstPossibleTxTime(tr,TreeDepth)
h = (TreeDepth - tr) * mu;
function h = firstRxTime(tr,TreeDepth)
h = (TreeDepth - tr - 1) * mu;


function b = moreToSendInSubTree(x,te,ot,on)
% Returns true if there are some nodes with data down the tree
if nargin == 3
	on = cell(1,te.N);
end
b = ~isempty([ot{x} on{x}]);
for z = te.ch{x}
	b = b + moreToSendInSubTree(z,te,ot,on);
end


function h = mu
%Period time in DMAC
h = 11e-3;
function h = nxtNatTxT( t, tr,TreeDepth )
h = nxtNatT( t, firstPossibleTxTime(tr,TreeDepth) );
function h = nxtNatRxT(t,tr,TreeDepth)
h = nxtNatT( t, firstRxTime(tr,TreeDepth) );
function h = nxtNatT( t, firstNatT)
h = firstNatT + ceil( (t - firstNatT) / compPer) * compPer;
function h = nxtIncTxT(t,tr,TreeDepth)
h = nxtIncT(t,nxtNatTxT(t,tr,TreeDepth));
function h = nxtIncRxT(t,tr,TreeDepth)
h = nxtIncT(t,nxtNatRxT(t,tr,TreeDepth));
function h = nxtIncT(t,nxtNatT)
sInc = nxtNatT - compPer;
h = sInc + ceil( (t - sInc) / TxPeriod) * TxPeriod;
function [r,s] = set(r,s,x,state,t)
% Compute the energy consumed
	switch state
		case {'Bckff', 'IdleState','Waiting4DataState','Wg4Ack'}
			pow = 'Idl';
		case { 'ReceivingDataState', 'ReceivingAckState'}
			pow = 'Rx';
		case {'TransmittingDataState', 'TransmittingAckState'}
			pow = 'Tx';
		case {'SleepState'}
			pow = 'Slp';
		otherwise
			error('I did not find the consumption for this state')
	end
	% Set new state
	s{x} = state;
	r{x}(t) = pow;
function [e,in,r,s,ixT] = txPkt(p,e,in,r,s,ixT,te,t)
% This transmitting function is independent of the data packet. A
% node, after concluding its reception, must check that it received
% the right kind of packet.
for kk = [te.txL{p.frm} te.ixL{p.frm}] % for all interfered nodes
	switch s{kk}
		case 'Waiting4DataState'
			[r,s] = set(r,s,kk,'ReceivingDataState',t);
			%Schedule 'endDataRx' event DataT later
			e = ev( e,t + DataT*1.001, kk, 'endDataRx');
			% Place the node into the input buffer
			in{kk} = [in{kk} p];
		case 'Wg4Ack'
			% Set 'ReceivingAckState'
			[r,s] = set(r,s,kk,'ReceivingAckState',t);
			%Schedule 'endAckRx' event AckT later
			e = ev(e, t + AckT *1.001, kk, 'endAckRx' );
			in{kk} = [in{kk} p];
		case 'Bckff'
			% remove 'startDataTx' event
			e = del(e,kk,'startDataTx');
			[r,s] = set(r,s,kk,'IdleState',t);
			% Do not schedule any event, as the node will stay in high energy
			% state until the the 'endCont' event
		otherwise
			ixT(kk) = t + p.drtn;	% WAITING FOR TRANSMISSION STATES
	end
end
function h = TxPeriod
%Time between the beginning of two consecutive listening in the
%increased duty cycle in DMAC
h = 5 * mu;