function this = setR(this,x,newRadioState,t)
% Compute the energy consumed
switch newRadioState
	case {'Idl' 'Rx' 'Tx' 'Slp'}
	otherwise
		error('Incorrect radio type')
end
switch this.radioState{x}
	case 'Rx'
		pow = PowerReceive;
	case 'Tx'
		pow = PowerTransmit;
	case 'Idl'
		pow = PowerIdle;
	case 'Slp'
		pow = PowerSleep;
	otherwise
		error('I did not find the consumption for this state')
end
this.engy(x) = this.engy(x) + (t - this.radioT(x)) * pow;

% Set new state
this.radioState{x} = newRadioState;

% Record the time when the state started to compute the energy
% consumption.
this.radioT(x) = t;