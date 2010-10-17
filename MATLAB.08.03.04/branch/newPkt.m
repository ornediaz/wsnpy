function h = newPkt(frm,dst,drtn,ty,Nr,src) %{{{
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
% vim:foldmethod=marker:ts=4:sw=4:tw=76
