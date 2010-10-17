function [eL,t,x,ty] = get(eL)
eL.numEvents = eL.numEvents + 1;
if isempty(eL)
	error('Attempt to retrieve event from empty list')
elseif eL.numEvents > eL.maxEvents
	error('Maximum number of events exceeded')
end
[t,ind] = min(eL.time);
eL.time(ind) = [];

x = eL.node(ind);
eL.node(ind) = [];

ty = eL.type{ind};
eL.type(ind)=[];

if t < eL.tPrevious
	error('We are going back in time!')
elseif t > 30
	error('This simulation has taken too long')
end