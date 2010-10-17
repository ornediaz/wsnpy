function te = tree(varargin)
switch nargin
	case 0
		x = 4;
		y = 4;
		txRg = 1.5;
		ixRg = 3;
		P = [ 0 0; 1 1; 2 2; 2 3.3; 3.3 2];
		te = tree([x y txRg ixRg],P);
	case 1
		if isa(varargin{1},tree)
			te = varargin{1};
		else
			error('Wrong argument type')
		end
	case 2
		if sum(size(varargin{1}) == [1 1]) == 2
			% (N,[x y txRg ixRg])
			N = varargin{1};
			if sum(size(varargin{2})== [1 4]) ~=2
				error('Incorrect size')
			end
			x = varargin{2}(1);
			y = varargin{2}(2);
			P=[0 0 ; rand(N-1,2) .* repmat([x y],N-1,1)];
			te = tree(varargin{2},P);
		elseif sum(size(varargin{1}) == [1 4]) == 2
			% ([x y txRg ixRg],P)
			te.x = varargin{1}(1);
			te.y = varargin{1}(2);
			te.txRg = varargin{1}(3);
			te.ixRg = varargin{1}(4);

			if size(varargin{2},2) ~= 2
				error('P should have two columns')
			end
			te.P = varargin{2};

			te.N = length(te.P);
			te.pt = dijkstra(te.P,te.txRg);
			te.py = ones(1,te.N);
			[te.tr,te.ch] = updateTierChildren(te.pt);
			[te.txL,te.ixL,te.up] = neigh(te.P,te.txRg,te.ixRg,te.tr);
			te.src = [];
			te.nPkts = 0;
			te = class(te,'tree');
		else
			error('Incorrect input arguments')
		end
	otherwise
		error('Incorrect number of inputs')
end