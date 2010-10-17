function p = agg(packets,x,y,alpha)
% Aggregate packets. Set source, destination duration.
% alpha is an aggregation coefficient between 0 (the size of a
% packet resulting from aggregation of many packets is the same as
% the size of one packete) and 1 (there is no size reduction).
if var([packets.Nr]) > 1e-3
	error('All packets should be from the same round')
end
drtn = DataT * (1+(length([packets.src])-1) * alpha);
p = nPkt(x,y,drtn,'DataK',packets(1).Nr,[packets.src]);