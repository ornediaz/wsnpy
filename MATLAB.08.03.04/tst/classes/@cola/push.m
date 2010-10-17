function p = push(p,k)
%q = p;
if (p.size == p.bufferSize)
	error('Queue is full and cannot accommodate another element')
end
p.buffer(p.tail) = k;
p.tail = mod(p.tail,p.bufferSize)+1;
p.size = p.size + 1;