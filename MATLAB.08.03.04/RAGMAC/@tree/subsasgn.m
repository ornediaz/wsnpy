function A = subsasgn(A,S,B)
switch S(1).type
	case '.'
		if length(S) == 1
			A.(S(1).subs) = B;
		elseif length(S) == 2
			switch S(2).type
				case '()'
					A.(S(1).subs)(S(2).subs{:}) = B;
				case '{}'
					A.(S(1).subs){S(2).subs{:}} = B;
				otherwise
					error('Unexpected type')
			end
		else 
			error('Unsupported number')
		end
	otherwise
		error('I was expecting a period')
end