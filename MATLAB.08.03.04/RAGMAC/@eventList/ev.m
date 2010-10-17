function eL = ev(eL,time, x, type)
% Add a new event to the event list
% switch type
% 	case {'endPktRx','endPktTx','inPktTx','startPktTx'}
% 	otherwise
% 		error('Unexpected event type')
% end
	
for w = 1:length(eL.time)
	if strcmp(eL.type(w),type) && eL.node(w) == x
		error('Attempt to insert duplicated event')
	end
end
eL.time(end+1) = time;
eL.node(end+1) = x;
eL.type{end+1} = type;

