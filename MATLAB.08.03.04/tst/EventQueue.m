function EventQueue
%eventList properties:
% * time
% * node
% * type

% This cannot be done efficiently, as we do not know where to place the event.
eventList = 
eventLisHeadPointer = 1;
eventListTailPointer = 1;

	function push(event)
		event
	end
	function event = pop
		
	end

end


function h = startContending,         h  = 1;end
function h = startWaiting4Data,          h  = 2;end
function h = startTxData,         h = 3; end

function h = EventListMaxSize, h =3; end