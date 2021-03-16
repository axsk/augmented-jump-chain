function [xi,yi,zi]=illustrate(q,f,n)

lx = max(q(1,:))-min(q(1,:));
ly = max(q(2,:))-min(q(2,:));
x=min(q(1,:)):lx/n:max(q(1,:));
y=min(q(2,:)):ly/n:max(q(2,:));
s=[kron(x,ones(1,length(x)));kron(ones(1,length(y)),y)];
[xi,yi,zi]=griddata(q(1,:),q(2,:),f,...
    x'*ones(1,length(x)),ones(length(y),1)*y);
surf(xi(:,1),yi(1,:),zi);
xi=xi(:,1);
yi=yi(1,:);