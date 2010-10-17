function h = FxdBkffT
% Fixed back off time, i.e., minimum time a node may back off
h = 4e-4;
if h <=B4AckT
	% This condition makes collisions of ACK with DATA unlikely
	error('The following does not hold: FxdBckffT > B4AckT')
end
