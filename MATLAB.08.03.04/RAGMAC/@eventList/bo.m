function eL= bo(eL,x,t)
% eL = bo(eL,x,t) reschedules a inPktTx from x if any. It does not
% issue an error if there is no such event.
found = f(eL,x,'inPktTx');
if length(found) > 1
	error('A node has more than one inPktTx in event list')
elseif length(found) == 1
	eL.time(found) = max(eL.time(found),t);
end