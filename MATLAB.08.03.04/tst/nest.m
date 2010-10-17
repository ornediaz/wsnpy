function nest
clc,format compact
x= 3;
inNest1(x);
inNest2(x);
x, y, whos
	function b =inNest1(b)
	y = 2* b;
	b = b* 2;
	end
	function b = inNest2(b)
	y = 3 * b;
	b = b* 3;
	end
end
