function b = subsref(a,index)
%SUBSREF Define field name indexing for asset objects
switch index.type
case '()'
   switch index.subs{:}
   case 1
      b = a.descriptor;
   case 2
      b = a.date;
   case 3
      b = a.currentValue;
   otherwise
      error('Index out of range')
   end
case '.'
   switch index.subs
   case 'descriptor'
      b = a.descriptor;
   case 'date'
      b = a.date;
   case 'currentValue'
      b = a.currentValue;
   otherwise
      error('Invalid field name')
   end
case '{}'
   error('Cell array indexing not supported by asset objects')
end