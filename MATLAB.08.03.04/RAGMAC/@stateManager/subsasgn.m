function a = subsasgn(a,index,newState)
switch index.type
	case '{}'
		switch newState
			case 'Lk4Ch'
			case 'Lk4Pt'
			case 'TxSt'
			case 'Wt4D'
			case 'SlpSt'
			otherwise
				error ('Unrecognized event type')
		end
		a.states{index.subs{:}} = newState;
end