function [p,d] = pop(p)
%q = p;
if (p.size == 0)
	error('Attempt to extract element from empty queue')
end
d = front(p);
p.head = mod(p.head,p.bufferSize)+1;
p.size = p.size - 1;