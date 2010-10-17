function a = subsasgn(a,index,val)
% SUBSASGN Define index assignment for asset objects
switch index.type
case '()'
   switch index.subs{:}
   case 1
      a.descriptor = val;
   case 2
      a.date = val;
   case 3
      a.currentValue = val;
   otherwise
      error('Index out of bounds')
   end
case '.'
   switch index.subs
   case 'descriptor'
      a.descriptor = val;
   case 'date'
      a.date = val;
   case 'currentValue'
      a.currentValue = val;
   otherwise
      error('Invalid field name')
   end
end