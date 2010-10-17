function testVarargin
pkt1 = createPacket(1,2,3,4)
pkt2 = createDataPacket(1,2,3,4,5,6)
end


function pkt = createPacket(varargin)
if nargin ~= 4
	error('Not enough input parameters')
end
pkt =struct(...
	'transmitter',varargin{1},...
	'destination',varargin{2},...
	'duration',varargin{3},...
	'pktType',varargin{4});
end

function pkt = createDataPacket(varargin)
if nargin ~= 6
	error('Not enough input parameters')
end
pkt = createPacket(varargin(1:4));
pkt.pktNr = varargin{5};
pkt.source = varargin{6};
end