function a = stateManager(varargin)
switch nargin
	case 0
		error('You need to pass one parameter at least')
	case 1
		if isa(varargin{1},'stateManager')
			a = varargin{1};
		else
			N = varargin{1};
			a.states = cell(1,N);
			for k = 1:N
				a.states{k} = 'Slp';
			end
			a = class(a,'stateManager');
		end
end
			