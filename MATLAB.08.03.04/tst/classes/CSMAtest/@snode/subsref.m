function b = subsref(a,index)
%SUBSREF Define field name indexing for asset objects
switch index.type
case '.'
   switch index.subs
	   case 'tier'
		   b = a.tier;
	   case 'listeningT'
		   b = a.listeningT;
	   otherwise
		   error('Invalid field name')
   end
	case '{}'
		error('Cell array indexing not supported by asset objects')
end