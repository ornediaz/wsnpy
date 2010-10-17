function te = prune(te)
% Prune nodes not intervening in tx
needed = ones(1,te.N) < .5;
for k = te.src
	q = k;
	while q > 0
		needed(q) = true;
		q = te.pt(q);
	end
end
te.pt(~needed) = -1;
[te.tr,te.ch] = updateTierChildren(te.pt);