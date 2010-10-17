function b = subsref(a,s)
switch s.type
	case '{}'
		b = a.states{s.subs{:}};
	otherwise
		error('Operation not supported by subsref')
end