function a = subsasgn(a,index,val)
% SUBSASGN Define index assignment for asset objects
switch index.type
case '.'
   switch index.subs
   case 'tier'
      a.tier = val;
   case 'parent'
      a.parent = val;
   otherwise
      error('Invalid field name')
   end
end