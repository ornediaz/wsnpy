function hand
h = struct('fun',@sin);
h.fun(pi/6)
h.fun() == @sin()


end