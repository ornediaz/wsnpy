function eL = eventList(a)
if nargin == 0
   eL.time = [];
   eL.node = [];
   eL.type = {};
	eL.tPrevious  = 0;
	eL.maxEvents = 1e4;
	eL.numEvents = 0;
   eL = class(eL,'eventList');
elseif nargin == 1 && isa(a,'eventList')
   eL = a;
elseif nargin ==1
	eL = eventList();
	eL.maxEvents = a;
end