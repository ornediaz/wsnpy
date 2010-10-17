function a = cola(varargin)
% Constructor function for cola object
% a = cola(descriptor, type, currentValue)
switch nargin
	case 0
		a = cola(6);
	case 1
		% if single argument of class cola, return it
		if (isa(varargin{1},'cola'))
			a = varargin{1};
		elseif (isa(varargin{1},'double'))
			a.bufferSize = varargin{1};
			a.buffer = 1:varargin{1};
			a.head = 1;
			a.tail = 1;
			a.size = 0;
			a = class(a,'cola');
		else
			error('Wrong argument type')
		end
	otherwise
		error('Wrong number of input arguments')
end