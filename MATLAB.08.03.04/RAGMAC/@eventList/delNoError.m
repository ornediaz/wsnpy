function eL = delNoError(eL,x)
% Delete any event from a certain node without without error
found = [];
for kk=1:length(eL.node)
	if eL.node(kk) == x
		found = [found kk];
	end
end
eL.time(found) = [];
eL.node(found) = [];
eL.type(found) = [];