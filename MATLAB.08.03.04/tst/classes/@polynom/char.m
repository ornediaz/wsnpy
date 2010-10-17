function s = char(p) 
% POLYNOM/CHAR   
% CHAR(p) is the string representation of p.c
if all(p.c == 0)
   s = '0';
else
   d = length(p.c) - 1;
   s = [];
   for a = p.c;
      if a ~= 0;
         if ~isempty(s)
            if a > 0
               s = [s ' + '];
            else
               s = [s ' - '];
               a = -a;
            end
         end
         if a ~= 1 | d == 0
            s = [s num2str(a)];
            if d > 0
               s = [s '*'];
            end
         end
         if d >= 2
            s = [s 'x^' int2str(d)];
         elseif d == 1
            s = [s 'x'];
         end
      end
      d = d - 1;
   end
end