function vec = getQueue(q)
if (q.size == 0)
	vec = [];
elseif (q.head < q.tail);
	vec = q.buffer(q.head:q.tail-1);
elseif (q.head >= q.tail)
	vec = q.buffer([q.head:end 1:q.tail-1]);
end