function te = genData(te,nS,nPkts)
% Gen traffic in nS points furthest frm sink.
% Input:
% => te: tree
% => nS: number of sources 
% => nPkts
% Output:
% => BOwn: buffer with the data


% store them in vector v
% FIXME: I should select the closest nodes to one point (the
% north-east corner), not the furthest nodes to another point (the
% south-west corner)
dist = sum( abs(te.P - repmat([te.x te.y],te.N,1)).^2 , 2 );
if sum(te.pt > 0) < nS
	error('There fewer connected nodes than desired sources')
end
[b ix] = sort( dist(:) + (te.x + te. y) *(te.pt(:) < 0) );
te.src = ix( 1 : nS );
te.nPkts = nPkts;