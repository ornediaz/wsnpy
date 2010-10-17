function t = boT(x,prty,trial)
% Random backoff time which depends on the priority.
if nargin == 0
	xtnd = 0;
	trial = 0;
elseif nargin == 2
	xtnd = (prty(x)-1) * .2;
	trial = 0;
elseif nargin == 3
	xtnd = (prty(x)-1) * .2;
elseif nargin > 3
	error('Incorrect number of inputs')
end
t = FxdBkffT + ContentionT * 2^trial * (rand + xtnd );
