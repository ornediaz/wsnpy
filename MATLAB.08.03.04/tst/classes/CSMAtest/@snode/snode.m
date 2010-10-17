function p = snode(varargin)
% ASSET Constructor function for snode object
% a = snode(descriptor, type, currentValue)
if nargin == 1
   if isa(varargin{1},'snode')
	   p = varargin{1};
   else
	   p.ID = varargin{1};
	   p.parent = 3;
	   p.tier = 0;
	   p.inBuff = []
	   p.outBuff = [];
	   p.listeningT = -1;
	   p.transmitT = -1;
	   p.energyConsumed = 0;
	   p = class(p,'snode');
   end
end