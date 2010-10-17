function [engy,dly] = orneMAC(...
	N,tier,upTierID,trnRng,interfRng,BOwn,P,xSide,ySide,verbose,kt,zt) 
%

BIn = cell(1,N);
BOut = cell(1,N); % Packets to be sent. FIFO.
chld = cell(1,N);
dly = -1;
engy = zeros(1,N); % Energy consumed
interfEndT = zeros(1,N); % End of interference
prnt = -ones(1,N);%Parent
prty = ones(1,N); % Priority relative to children. If 1, there are no
% siblings precede it
radioT = zeros(1,N); % Time when current state started
radioSt = cell(1,N);for z=1:N,radioSt{z}='Slp';end
rd = ones(1,N); % Packet from me that my parent expects
st = cell(1,N); for z =1:N,st{z}='SlpSt';end
upTierNoC = cell(1,N);% The numberOfChildren in the upper tiers


%% Create traffic and initial events 
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
%% Event loop %{{{
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
	if dly > 0 
		return
	end
	if verbose
		disp('****************')
		disp('Current event')
		fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,e.type);
		disp('Current event list')
		dispEvntL
		fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,e.type);
	end
	% 
	for bb = 1: N
		if length(chld{bb}) == 1 && strcmp(st{bb},'Wt4D') ...
				&& rd(bb) ~= rd(chld{bb}) 
			error('Only one child and waiting for data, but child ignores it')
		elseif length(findEvent(bb,'inPktTx') ) > length(BOut{bb})
			error('Too many inPktTx events')
		end
	end
	if kt == 3 && zt == 2 && t > 1
		disp('Something wrong')
	end
	

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
					% FIXME: handle the case of replicates. If one
					% receives a replicate, discard it
	
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
				if verbose
					plt(P,prnt,xSide,ySide)
				end
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
			rd(x) = 1;
			setState(x,'TxSt');
			%}}}
		case 'stWt4D' %{{{
			% We only execute this after the setup phase.
			rd(x) = 1;
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
end %while }}}
%% NESTED FUNCTIONS 
	function ackChld(x,Nr) %{{{
	for y = chld{x}
		rd(y)= Nr + 1;
		% My children can initiate the next rd.
		if length(BOut{y}) > 1
			error('length(BOut) should be smaller than 1')
			% Cancel retx of any packet whose Nr is smaller than Nr
		elseif length(BOut{y}) == 1 && strcmp(BOut{y}.ty,'DataK') ...
				&& BOut{y}.Nr==Nr
			% delete it
			if ~strcmp(radioSt{y},'Tx') && ~strcmp(radioSt{y},'Slp')
				removeEvent(y,'inPktTx');
				BOut{y} = [];
			else %if strcmp(radio
				% FIXME
				disp('Something suspicious is going on.')
				BOut{y} = [];
			end
		end

		if ~moreToSendInSubTree(y,chld,BOut,BOwn)
			setState(y,'SlpSt')
		elseif isempty(chld{y}) && isempty(BOut{y})
			% No children but something in BOwn.
			% The node is a source that does not need to merge the data.
			BOut{y} = BOwn{y}(1);
			BOwn{y}(1) = [];
			setState(y,'TxSt');
		elseif ~isempty(BOut{y}) && strcmp(st{y},'TxSt')
			%It has a packet available for transmission
			% The node has children
			setState(y,'TxSt');
			% With this code, a new transmission will be started if needed.
		elseif strcmp(st{y},'TxSt')
			%Its output buffer is empty and it needs to get data from children
			setState(y,'Wt4D')
			ackChld(y,Nr);
		elseif strcmp(st{y},'Wt4D')
			if sum(rd(chld{y}) == rd(y) ) < 0
				error('I am waiting, but no one is aware of that')
			end
		else
			error('We should not arrive here')
		end
	end
	for w = find(prnt == 1);
		if ~moreToSendInSubTree(w,chld,BOut,BOwn)
			dly = t;
			return
		end
	end
	end %}}}

	function ackRx(x,Nr)
	rd(x) = Nr + 1;
	if length(BOut{x}) ~= 1
		error('length(BOut) should be 1')
	elseif ~strcmp(BOut{x}.ty,'DataK')
		error('There should be a DataK pkt in BOut')
	end
	% execute the code below only if there is a packet with
	% number = than current
	if BOut{x}.Nr == Nr
		BOut{x}= [];
		removeEvent(x,'inPktTx');
		% FIXME: the line below is incorrect
		if ~moreToSendInSubTree(x,chld,BOut,BOwn)
			setState(x,'SlpSt')
		elseif isempty(chld{x}) && isempty(BOut{x})
			% Nko children but something in BOwn.
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
	elseif strcmp(st{x},'TxSt')
		setState(x,'TxSt')
	end
	end
	function t = backoffT(x) %{{{
	t = FxdBkffT + ContentionT* (1 + ( prty(x) - 1 ) * .2 ) * rand;
	end %}}}
	function evnt(time, x, type) %{{{
	for w = 1:length(evntL)
		if strcmp(evntL(w).type,type) && evntL(w).node == x
			if time < evntL(w).time
				evntL(w).time = evntL(w).time + 3 * B4AckT + 3 * BOut{x}(1).drtn;
			end
			%error('Duplicate event')
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
		evnt( t + p.drtn * 1.001, y, 'endPktRx' )
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
			||  strcmp(st{y},'TxSt') ...
			&& backing(y,BOut,rd) && (bo || strcmp(radioSt{y},'Idl') )
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
						if verbose
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
						%Cannot initiate the next rd. Must wait 4 my parent to rx Ack
						%from its parent.
						ackRx(x,p.Nr)
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
			if p.dst == x && strcmp(p.ty,'DataK')
				sendAck(x,p)
				if  p.Nr < rd(x)
					BIn{x}(end) = [];
				end
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
	found = findEvent(x,type);
	if length(found) >1
		error('I found too many elements to remove')
	elseif isempty(found)
		error('I found no events to remove')
	end
	evntL(found) = [];
	end %}}}
	function found = findEvent(x,type) %{{{
	found = [];
	for kk=1:length(evntL)
		if evntL(kk).node == x && strcmp(evntL(kk).type,type)
			found = [found kk];
		end
	end
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
			if backing(x,BOut,rd)
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
if dly < 0
	error('No delay computed')
end
end % END OF orneMAC
function b = backing(y,BOut,rd)
b = ~isempty(BOut{y}) ...
	&& ~isempty(BOut{y}.Nr) ...
	&& BOut{y}(1).Nr == rd(y);
end
function tc = endTierT(t) %{{{
% Time when the current tier setup phase ends
t0 = floor(t/CompletePeriod) * CompletePeriod;
tc = t0 + ceil( (t-t0) / TierStpT )*TierStpT;
end %}}}
function h = nxtL4Pt(tier,TreeDepth,t)%{{{
t0 = floor(t/CompletePeriod) * CompletePeriod;
h = t0 + (TreeDepth - tier) * TierStpT;
end %}}}
function h = nxtL4Ch(tier,TreeDepth,t) %{{{
t0 = floor(t/CompletePeriod) * CompletePeriod;
h = t0 + (TreeDepth - tier - 1) * TierStpT;
end %}}}
