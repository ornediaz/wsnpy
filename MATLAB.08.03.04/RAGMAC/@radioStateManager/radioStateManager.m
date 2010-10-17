function this = radioStateManager(N)
switch nargin
	case 0 
		this = radioStateManager(5);
	case 1
		this.engy = zeros(1,N); % Energy consumed
		this.radioT = zeros(1,N); % Current radio-state start time
		this.radioState = cell(1,N);
		for z=1:N
			this.radioState{z}='Idl';
		end % Radio state
		this = class(this,'radioStateManager');
	otherwise
		error('Incorrect number of inputs')
end
