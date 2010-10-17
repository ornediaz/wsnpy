function transmitPkt(pkt,interfLst)
global nL
% This function 

% Deliver the packet to its destination.
nL{pkt.dest} = transmit(nL{pkt.dest});

% Send as interference to all nodes within interference range of the
% source.
for node = interfLst{pkt.src}

	%nL{node} = interf(nL{node},pkt);
end