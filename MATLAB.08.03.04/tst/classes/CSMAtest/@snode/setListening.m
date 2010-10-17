function a = setListening(a)
global mu
global numTiers
global t

firstList  = (numTiers - a.tier - 1) * mu;
period = (numTiers - 1) * mu;
a.listeningT = firstList + ceil(t/period) * period;

if a.tier == numTiers
	% Nodes in the outermost tier do not need to listen for nodes in the
	% next tier, as there is none. To indicate this property,
	% we set the listening time to be negative;s
	a.listeningT = 1;
end
