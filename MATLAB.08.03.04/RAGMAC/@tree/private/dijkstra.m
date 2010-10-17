function [previous] = dijkstra(P,txRg)
N = length(P);
previous=-1*ones(1,N);
% It is necessary to use a finite value for Inf! I have checked that
% Matlab's Inf does not work in my algorithm.
Inf = 1e5; 
d=ones(1,N)*Inf;
d(1)=0;
% If element 'i' of temp is zero, it means it has not been processed
% yet
temp=zeros(1,N);
for kk=1:N %For all vertices
	[m u]=min(d+temp*2*Inf);
	temp(u)=1;
	for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
		dist=norm(P(u,:)-P(jj,:)); % distance to 'jj' from 'u'
		if dist > txRg
			dist=Inf;
 		else
 			dist = 1;
		end
		alt=d(u)+dist;
		if (alt < d(jj))
			d(jj) = alt;
			previous(jj) = u;
		end
	end
end