function [engy,t] = txPhase(te,ac,vrb)
%Tx phase without ack loss and without stop and go
% 
% This simulation does not consider some phenomena that would occur in
% reality:
% * The lost of ACK packets causes retransmissions, which cause more
% contention and deteriorate performance.
%
% The protocol is not stop and go. This means that a node that did
% not transmit packets from one round to its parent may be receiving
% packets from the previous round from its children. This is easier
% to implement in a real system because we do not need to propagate
% information beyond children. In fact, the system does not require
% packet exchange besides unnecessary retransmissions and the
% corresponding acknowledgements. Perhaps stop and go would avoid
% some collisions, but it also introduces overhead and may not use
% the channel as much as possible. Therefore, I am not sure which
% one would perform better.
%
% I will use this transmission phase in my algorithm, but there is
% no significant novelty in it. It is possible that there are better
% algorithms. I will use this function also with other protocols:
% Shortest Path Tree and Steiner Tree.
%
% This protocol uses CSMA. Nodes are awake all the time and thus the
% energy consumption will be very high. DMAC outperforms this
% protocol several times in terms of energy.
%
% Inputs:
% * te.N: number of nodes
% * te.tr: tr number of the nodes
% * te.pt: parent of each node
% * te.py: priority with respect to siblings, 1 is highest
% * te.ch: children of each node, Used to know 4 how many pkts 2 wait
% * te.txL: nodes within transmission range
% * te.ixL: nodes within interference range
% * te.src
% * te.nPkts
% * ac: aggregation coefficient, between 0 and 1
% * vrb: if true, it shows information useful for debugging
% 
% Outputs:
% * engy: energy consumed
% * t: simulated time to tx all the pkts

ack = zeros(1,te.N); % Last packet acknowledged by parent
e = eventList(1e5);
in = cell(1,te.N); % Input buffer
ixEndT = zeros(1,te.N); % End of interference
ot = cell(1,te.N); % Packets to be sent. FIFO.
on = cell(1,te.N);
r = radioStateManager(te.N);
t = 0;
xpt = ones(1,te.N); % Expected packet number
trial = zeros(1,te.N);



for k = te.src(:)'
	aux = repmat(nPkt(k,te.pt(k),DataT,'DataK',1,k),te.nPkts,1);
	for gg = 1:te.nPkts % Packet number
		% Append the packet in the output buffer
		aux(gg).Nr = gg;
	end
	if isempty(te.ch{k})
		%Schedule tx of nodes with data and w/o children
		e = ev(e , t + boT(k,te.py) , k , 'inPktTx');
		ot{k} = aux;
	else
		on{k} = aux;		
	end
end

% Initialize the radio state of all nodes
for k = 1:te.N
	if sum(te.src == k) || ~isempty(te.ch{k})
		r{k}(t) = 'Idl';
	else
		r{k}(t) = 'Slp';
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
	switch ty
		case 'endPktRx'
			p = in{x}(end);
			r{x}(t) = 'Idl';
			if ixEndT(x) > t - p.drtn || p.dst ~=  x
				% Unsuccessful reception due to interference
				in{x}(end) = [];
			elseif p.dst == x
				%Acknowledge packet transmission
				ot{p.frm}(1) = [];
				e = del(e,p.frm,'inPktTx');
				ack(p.frm) = ack(p.frm) + 1;
				if length(in{x}) == length(te.ch{x})
					%Announce children that I expect the next
					%packet.
					xpt(x) = xpt(x)+ 1;
					% Create list of all packets to be
					% aggregated.
					pkts = in{x};
					if ~isempty(on{x})
						pkts = [pkts on{x}(1)];
						on{x}(1) = []; %Remove it
					end
					ot{x} = [ot{x} agg(pkts,x,te.pt(x),ac)];
					in{x} = [];
					for z = [x te.ch{x}]
						if te.tr(z) ...
								&& xpt(te.pt(z)) > ack(z)...
								&& isempty(f(e,z,'inPktTx'))...%%&& strcmp(r{z},'Idl') ...
								&& ~isempty(ot{z})
							% There should not be any requirement regarding
							% the radio state, otherwise the program fails.
							% If I impose z to be Idle and the parent
							% receives the packet before the siblings then
							% the siblings will be still in Rx state.
							e = ev( e, t+boT(z,te.py), z, 'inPktTx' );
							trial(z) = 0;
						end
					end
				end
			end
		case 'endPktTx'
			%Retransmit packet until the 
			trial(x) = trial (x) + 1;
			e = ev(e, max(t,ixEndT(x)) + boT(k,te.py,trial(x)),x,'inPktTx');
			r{x}(t) = 'Idl';
		case 'inPktTx' % Initiate packet transmission
			% It considers the delay involved in carrier sensing.
			r{x}(t) = 'Slp';
			% A node whose radio is in SlpSt will not cancel its
			e = ev(e, t + CarrSensT, x, 'startPktTx' );
		case 'startPktTx'
			p = ot{x}(1);
			r{x}(t) = 'Tx';
			e = ev(e, t+p.drtn * .999,x,'endPktTx');
			%For nodes within interference range
			for y = [ te.ixL{x} te.txL{x} ]
				if sum( y == te.txL{x} ) ...
						&& strcmp(r{y},'Idl') ...
						&& ixEndT(y) < t % (no interference)
					% Place the packet into the input buffer.
					in{y} = [in{y} p];
					r{y}(t) = 'Rx';
					e = ev(e, t + p.drtn * 1.001, y, 'endPktRx' );
					% FIXME The line above fails sometimes, because the
					% node y is already receiving. But if the node was
					% already receiving, why was the state 'Idl'
				elseif t + p.drtn > ixEndT(y)
					% If new interference finishes later than
					% previous one, readjust the end of
					% interference time
					ixEndT(y) = t + p.drtn;
				end
				% Handle nodes backing off
				eventEnd = max( ixEndT(y), t + p.drtn );
				e = bo(e,y,eventEnd + boT(y,te.py));
			end
		otherwise
			error('Unexpected event: %s',ty)
	end %switch ty
end %while
engy = getE(r);
for q = 1:te.N
	if te.tr(q) && (~isempty(on{q}) || ~isempty(ot{q}))
		error('There were undelivered messages')
	end
end