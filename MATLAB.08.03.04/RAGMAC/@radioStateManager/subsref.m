function b = subsref(a,s)
% If we type a{2}, I want to return a.radioState(2)
switch s.type
	case '{}'
		b = a.radioState{s.subs{:}};
end
	