function ab
clc
persistent q
x=7;
ba(x);
q
ba(x);
q
	function y = ba(z)
		if q
			q = q * 2;
		else
			q = 2 * z;
		end
		y = z;
	end
end