function a = set(a,varargin)
% SET Set asset properties and return the updated object
propertyArgIn = varargin;
while length(propertyArgIn) >= 2,
   prop = propertyArgIn{1};
   val = propertyArgIn{2};
   propertyArgIn = propertyArgIn(3:end);
   switch prop
   case 'Descriptor'
      a.descriptor = val;
   case 'Date'
      a.date = val;
   case 'CurrentValue'
      a.currentValue = val;
   otherwise
      error('Asset properties: Descriptor, Date, CurrentValue')
   end
end