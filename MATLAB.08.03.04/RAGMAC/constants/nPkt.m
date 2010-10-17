function h = nPkt(frm,dst,drtn,ty,Nr,src)
% New pkt using inputs: frm,dst,drtn,ty,Nr,src
switch ty
	case {'RelReqK','RelOffK','DataK','AckK'}
		h = struct(...
			'frm',frm,...
			'dst',dst,... % destination
			'drtn',drtn,... %duration
			'ty',ty,...
			'Nr', Nr,...
			'src',src);
	otherwise
		error('Unexpected packet type')
end %switch
