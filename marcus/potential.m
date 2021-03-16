function [val]=potential(x_in)

x=x_in(1);
y=x_in(2);

val = 3*exp(-x^2-(y-1/3)^2)-3*exp(-x^2-(y-5/3)^2)-5*exp(-(x-1)^2-y^2);
val = val-5*exp(-(x+1)^2-y^2)+ 0.2*x^4+0.2*(y-1/3)^4;
% val = exp(- 3.34 * val);