function  te = steiner(te,root)
% Appr Steiner tree (Dijkstra + my intuition)
% In a paper they use Prim's algorithm, which I believe is similar
% to Dijtra's.
% prev(i) == -1 if the node does not belong to the Steiner tree.
if nargin == 1
	root = 1;
end
prev = sdijkstra(te.P,te.txRg,te.src,root);
cont = true;
while cont
	% Delete non-terminals of degree 1 from the spanning tree.
	cont = false;
	for z = 1:te.N
		if sum(z == te.src) < 1 % It is not a source
			if prev(z) ~= -1 && sum(prev == z) < 1
				% No one depends on it
				prev(z) = -1;
				cont = true;
			end
		end
	end
end
te.pt = prev;
[te.tr,te.ch] = updateTierChildren(te.pt);

