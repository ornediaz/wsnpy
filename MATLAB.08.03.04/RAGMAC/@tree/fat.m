function [en,t,te,nerr] = fat(te,vrb)
% Returns the aggregation tree obtained with FAT
% Input
% => tr: tier
% Output
% => en: energy
% => dy: delay
% => te: modified fields: pt,ch,py
%
% FIXME: there was a situation in which a node without children or
% data had a node as a parent. 
%
in = cell(1,te.N); %input buffer
ot = cell(1,te.N); % Output buffer. Packets to be sent. FIFO.
ch = cell(1,te.N); %Children
nerr = 0;

ixEndT = zeros(1,te.N); % End of interference
pt = -ones(1,te.N);%Parent
py = ones(1,te.N);
% Priority relative to children. If 1, no siblings precede it
r = radioStateManager(te.N);
s = stateManager(te.N);
for z =1:te.N
	s{z}='SlpSt';
end
upN = cell(1,te.N);% The numberOfChildren in the upper tiers


%% Create traffic and initial events
t = 0;
e = eventList(1e5);
% Schedule initial events for all nodes with children
for z = 1: te.N
	if te.tr(z)~= max(te.tr)
		e = ev(e, nxtL4Ch(te.tr(z),max(te.tr),t) , z, 'stLk4Ch');
	elseif sum(z == te.src)
		e = ev(e, nxtL4Pt(te.tr(z),max(te.tr),t), z, 'stLk4Pt');
	end
end
%% Event loop %{{{
while ~isempty(e)
	[e,t,x,ty] = get(e);
	if vrb
		%disp('****************')
		%disp('Current event')
		fprintf('t = %7.5f, x = %3.0f, type = %s \n',t,x,ty);
		%display(e)
	end
	if strcmp(ty,'endPktRx')
		p = in{x}(end);
		in{x}(end) = [];
		r{x}(t) = 'Idl';
		if ixEndT(x) > t - p.drtn, continue % Ruined reception
		elseif strcmp(s{x},'Lk4Ch')
			if p.dst ~= x, continue
			elseif ~strcmp(p.ty,'RelReqK')
				error('I should not be receiving this packet')
			end
			% Reply with a RelOffK packet
			ot{x}=[nPkt(x,p.frm,RelOffT,'RelOffK',p.Nr,p.src) ot{x}];
			r{x}(t) = 'Slp'; % This way a node cannot start receiving
			e = ev(e, t + B4AckT , x,'inPktTx');
		elseif strcmp(s{x},'Lk4Pt')
			if ~strcmp(p.ty,'RelOffK') && ~strcmp(p.ty,'RelReqK')
				error('I should not be receiving this packet')
			elseif ~strcmp(p.ty,'RelOffK'), continue
			elseif p.dst == x
				if vrb
					disp('==========================')
					fprintf('Node %d found % d as parent\n',x,p.frm)
					disp('==========================')
				end
				% I have obtained a parent.
				pt(x) = p.frm;
				ch{p.frm}(end+1) = x;
				% Cancel retransmission
				e = del(e,x,'inPktTx');
				ot{x} = [];
				r{x}(t) = 'Slp';
			else
				% Update next tier's number of children
				ii = find ( te.up{x} == p.frm );
				upN{x}(ii) = upN{x}(ii) + 1;
				% Set relay with more children as RelReq destination
				[C ii] = max( upN{x} );
				ot{x}(1).dst = te.up{x}(ii);
			end
		end
	elseif strcmp(ty,'endPktTx')
		r{x}(t) = 'Idl';
		if strcmp(ot{x}(1).ty,'RelReqK')
				% Retransmit until ACK reception.
				e = ev(e, max(t,ixEndT(x)) + boT, x, 'inPktTx');
		elseif strcmp(ot{x}(1).ty,'RelOffK')
			% Remove RelOff pkt from output buffer.
			ot{x}(1) = [];
		else
			error('Unexpected packet type')
		end
	elseif strcmp(ty,'inPktTx') 
		r{x}(t) = 'Slp';
		e = ev( e, t + CarrSensT, x, 'startPktTx' );
	elseif strcmp(ty,'stLk4Ch')
		r{x}(t) = 'Idl';
		s{x} = 'Lk4Ch';
		ch{x} = [];
		e = ev( e, t + TierStpT, x , 'stLk4Pt' );
	elseif strcmp(ty,'stLk4Pt')
		if strcmp(s{x},'Lk4Ch')&& (sum(x==te.src)<1) && isempty(ch{x})
			% No data nor children, thus no point in looking 4 parent
			s{x} = 'SlpSt';
			r{x}(t) = 'Slp';
		elseif ~te.tr(x) % If we are in te.tr 0
			s{x} = 'Wt4D';
			r{x}(t) = 'Idl';
		else % We need to look for a parent
			% Tell my children their priority
			for z = 1:length(ch{x})
				py(ch{x}(z)) = z;
			end
			s{x} = 'Lk4Pt';
			r{x}(t) = 'Idl';
			% Clear parent
			pt(x) = -1;
			% Clear the count of next tier neighbor's children.
			upN{x} = zeros(size(te.up{x}));
			% Create the new relay request packet
			ot{x} = nPkt(x,te.up{x}(1),RelReqT,'RelReqK',[],[]);
			% Schedule the transmission
			e = ev(e, t + boT, x, 'inPktTx' );
		end
	elseif strcmp(ty,'startPktTx')
		p = ot{x}(1);
		r{x}(t) = 'Tx';
		e = ev(e, t + p.drtn, p.frm, 'endPktTx');
		%For nodes within interference range
		for y = [te.txL{x} te.ixL{x}]
			% If within transmission range and idle, make them Rx.
			if sum(y==te.txL{x}) && strcmp(r{y},'Idl') && (ixEndT(y)<t)
				% Place the packet into the input buffer.
				in{y} = [in{y} p];
				r{y}(t) = 'Rx';
				e = ev(e, t + p.drtn * 1.01, y, 'endPktRx' );
			elseif t + p.drtn > ixEndT(y)
				% If new interference finishes later than previous one,
				% readjust the end of interference time
				ixEndT(y) = t + p.drtn;
			end
			% Handle nodes backing off
			eventEnd = max( ixEndT(y), t + p.drtn );
			if strcmp(s{y},'Lk4Pt') && pt(y) < 0 %Not found parent yet
				if eventEnd + NdStpT > endTierT(t)
					% No more time to look for a parent.
					%error('I did not find a parent')
					nerr = nerr + 1;
					pt(y) = ot{y}(1).dst;
					paux = ot{y}(1).dst;% new parent
					ch{paux}(end+1) = y;
					py(y) = length(ch{paux});
					% Cancel retransmission
					e = delNoError(e,y);
					ot{y} = [];
					r{y}(t) = 'Slp';
					
				else
					e = bo( e, y, eventEnd + boT );
				end
			end
		end
	else
		error('Unexpected event: %s',e.type)
	end
end %while }}}
te.pt = pt;
% TODO: introduce somehow here the prune function to account for the
% situation in which two nodes register the same node as a child.
[te.tr,te.ch] = updateTierChildren(te.pt);
en = getE(r);

function tc = endTierT(t)
% Time when the current te.tr setup phase ends
t0 = floor(t/compPer) * compPer;
tc = t0 + ceil( (t-t0) / TierStpT )*TierStpT;

function h = NdStpT
% Setup time per tier used in OrneMAC
h = FxdBkffT + 2* ContentionT + RelReqT + B4AckT + RelOffT;

function h = nxtL4Pt(tr,TreeDepth,t)%{{{
t0 = floor(t/compPer) * compPer;
h = t0 + (TreeDepth - tr) * TierStpT;

function h = nxtL4Ch(tr,TreeDepth,t) %{{{
t0 = floor(t/compPer) * compPer;
h = t0 + (TreeDepth - tr - 1) * TierStpT;

function h = TierStpT
% Maximum time during which nodes in a tier may look for a parent.
%NdStpT * nContend;
 h =  13 * NdStpT;
 