function b = moreToSendInSubTree(x,chld,BOut,BOwn) 
b = ~isempty([BOut{x} BOwn{x}]);
for z = chld{x}
	b = b + moreToSendInSubTree(z,chld,BOut,BOwn);
end
