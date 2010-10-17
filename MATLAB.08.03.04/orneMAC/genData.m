function BOwn = genData(P,p,upT,nS,nPkts)
% Input:
% => P: coordinates of the nodes. Two columns and N rows.
% => p: previous is 1xN and contains the parent
% => upT: cell(1,N) with IDs of nodes in upper tiers
% => nS: number of sources 
% Output:
% => BOwn: buffer with the data

BOwn = cell(1,length(P));

%% Get the indices of the nS nodes furthest from the point (0,0) and
% store them in vector v

[b ix] = sort( sum( abs(P).^2 , 2 ) - (p' < 0) * 9e9  );
v = ix( end-nS+1 : end );

% Generate data
for z = v'
	for gg = 1:nPkts; % Packet number
		% Append the packet in the output buffer
		BOwn{z} = [BOwn{z} newPkt(z,upT{z}(1),DataT,'DataK',gg,z)];
	end
end
