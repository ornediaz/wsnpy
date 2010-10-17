function [tier,upTierID,trnRng,interfRng,P,previous] = ...
	topology(N,txRg,ixRg,xSide,ySide)
interfRng = cell(1,N);
upTierID = cell(1,N);
trnRng = cell(1,N);

P=rand(N,2) .* repmat([xSide ySide],N,1);


INF=1000;
d=ones(1,N)*INF;
tier = Inf(1, N);
root = 1;
P(root,:) = [0 0];
d(root)=0;
tier(root) = 0;
previous=-1*ones(1,N);
% If element 'i' of temp is zero, it means it has not been processed yet
temp=zeros(1,N);
for kk=1:N %For all vertices
	[m u]=min(d+temp*2*INF);
	temp(u)=1;
	for jj=[1:u-1 u+1:N] % For each neighbor 'jj' of 'u'
		dist=norm(P(u,:)-P(jj,:)); % distance to 'jj' from 'u'
		if dist > txRg
			dist=INF;
		end
		alt=d(u)+dist;
		if (alt < d(jj))
			d(jj) = alt;
			previous(jj) = u;
		end
	end
end
% Compute each node's tier
for x=1:N
	p = x;
	n = 0;
	while p ~= -1 && p~=root
		n = n+1;
		p = previous(p);
	end
	if p ~= -1
		tier(x) = n;
	end
end
% if ~isempty(tier == Inf)
% 	error('Some nodes are in an infinite tier')
% end
for x = 1:N
	for y = [1:x-1 x+1:N]
		di = norm(P(x,:)-P(y,:));
		if di < txRg
			trnRng{x} = [trnRng{x} y];
			if tier(y) == tier(x) - 1
				upTierID{x} = [upTierID{x} y];
			end
		elseif di < ixRg
			interfRng{x} = [interfRng{x} y];
		end
	end
end
