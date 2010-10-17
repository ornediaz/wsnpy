function previous = sdijkstra(P,txRg,src,root)
% Super dijkstra algorithm that chooses carefully which minimum
N = length(P);
previous=-1*ones(1,N);
Inf = 25; % Essential! Matlab's Inf does not work in my algorithm.
d=ones(1,N)*Inf;
d(root)=0;
% If element temp(i) zero, 'i' has not been processed yet
temp=zeros(1,N);
for kk=1:N %For all vertices
	u = extract(d,temp,src,previous,Inf);
	temp(u)=1;
	for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
		dist=norm(P(u,:)-P(jj,:)); % distance to 'jj' from 'u'
		if dist > txRg
			dist=Inf;
		else
			dist = 1 - .1* sum(jj == src);
		end
		alt=d(u)+dist;
		if (alt < d(jj))
			d(jj) = alt;
			previous(jj) = u;
		end
	end
end


function u = extract(d,temp,src,previous,Inf)
% Give preference to:
% => sources
% => nodes with a higher degree
% =>
N = length(d);
v = d+temp*2*Inf;
m=min(v);
ind = find(v == m);
priority = zeros(1,N);
for z = ind
	priority(z) = 1e4 + sum(z == src) * 100 + sum(z == previous);
end
[k u] = max(priority);
