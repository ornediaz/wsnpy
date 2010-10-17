function [tr,ch] = updateTierChildren(pt)
N = length(pt);
tr = zeros(1,N);
ch = cell(1,N);
for z=1:N
	p = z;
	n = 0;
	while p ~= -1 && p~=1 %1 is the sink
		n = n+1;
		p = pt(p);
	end
	if p == 1
		tr(z) = n;
	end
	ch{z} = find( pt == z );
end
