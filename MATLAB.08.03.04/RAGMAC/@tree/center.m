function te = center(te)
pos = te.P(te.src,:);
ave = repmat(mean(pos),length(te.src),1);
[c,i] = min(sum(abs(pos - ave ).^2,2));
aggregator = te.src(i);
te = steiner(te,aggregator);

pt = sdijkstra(te.P,te.txRg,te.src,1);
q = aggregator;
while q ~=1
	te.pt(q) = pt(q);
	q = pt(q);
end
[te.tr,te.ch] = updateTierChildren(te.pt);

