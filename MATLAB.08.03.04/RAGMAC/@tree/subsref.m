function B = subsref(A,S)
switch S(1).type
	case '.'
		if length(S) == 1
			B = A.(S(1).subs);
		else
			switch S(2).type
				case '()'
					B = A.(S(1).subs)(S(2).subs{:});
				case '{}'
					if length(S) == 2
						B = A.(S(1).subs){S(2).subs{:}};
					elseif length(S) == 3
						switch S(3).type
							case '()'
								B = A.(S(1).subs){S(2).subs{:}}(S(3).subs{:});
							otherwise
								error('Referencing method not supported')
						end
					else
						error('Unexpected referencing')
					end
			end
		end
	otherwise
		error('I was expecting a period')
end