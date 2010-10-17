function found = f(eL,x,type)
% Returns indices of events of certain node and typ
found = [];
for kk=1:length(eL.time)
	if eL.node(kk) == x && strcmp(eL.type(kk),type)
		found = [found kk];
	end
end