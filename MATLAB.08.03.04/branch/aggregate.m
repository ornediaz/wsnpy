function p = aggregate(packets,x,y)
if var([packets.Nr]) > 1e-3
	error('All packets should be from the same round')
end
drtn = DataT * log2(1+length([packets.src]));
p = newPkt(x,y,drtn,'DataK',packets(1).Nr,[packets.src]);

