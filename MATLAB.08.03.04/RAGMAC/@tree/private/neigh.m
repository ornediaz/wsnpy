function [txL,ixL,up] = neigh(P,txRg,ixRg,tr)
N = length(P);
txL = cell(1,N);
ixL = cell(1,N);
up = cell(1,N);
for z = 1:N
	for y = [1:z-1 z+1:N]
		di = norm(P(z,:)-P(y,:));
		if di < txRg
			txL{z} = [txL{z} y];
			if tr(y) == tr(z) - 1
				up{z} = [up{z} y];
			end
		elseif di < ixRg
			ixL{z} = [ixL{z} y];
		end
	end
end
