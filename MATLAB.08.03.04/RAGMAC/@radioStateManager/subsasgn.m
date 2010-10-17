function a = subsasgn(a,index,val)
switch index(1).type
	case '{}'
		a = setR(a,index(1).subs{:},val,index(2).subs{:});
end