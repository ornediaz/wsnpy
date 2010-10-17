function a = asset(varargin)
% ASSET Constructor function for asset object
% a = asset(descriptor, type, currentValue)
switch nargin
case 0
% if no input arguments, create a default object
   a.descriptor = 'none';
   a.date = date;
   a.type = 'none';
   a.currentValue = 0;
   a = class(a,'asset');  
case 1
% if single argument of class asset, return it
   if (isa(varargin{1},'asset'))
      a = varargin{1};
   else
      error('Wrong argument type')
   end 
case 3
% create object using specified values
   a.descriptor = varargin{1};
   a.date = date;
   a.type = varargin{2};
   a.currentValue = varargin{3};
   a = class(a,'asset');
otherwise
   error('Wrong number of input arguments')
end