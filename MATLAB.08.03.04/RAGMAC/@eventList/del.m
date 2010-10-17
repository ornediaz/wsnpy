function eL = del(eL,x,type)
found = f(eL,x,type);
if length(found) > 1
	error('I found too many elements to remove')
elseif isempty(found)
	error('I found no events to remove')
end
eL.time(found) = [];
eL.node(found) = [];
eL.type(found) = [];