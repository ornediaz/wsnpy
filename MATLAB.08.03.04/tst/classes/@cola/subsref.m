function b = subsref(p,index)
% SUBSREF Define field name indexing for portfolio objects
switch index(1).type
case '()'
   vec=getQueue(p);
   b = vec(index(1).subs{1});
   if (p >= p.size)
	   error('Attempt to obtain an element number bigger than the queue size')
	otherwise
	error('The only type of subreferencing is with ()')
end