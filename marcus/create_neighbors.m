
function [points, weights, rate] = create_neighbors (point)
  beta=3.5;
  points=zeros(length(point),2*(length(point)-1));
  weights=zeros(1,2*(length(point)-1));
  for k=1:(length(point)-1)
    points(:,2*k-1)=point;
    points(k,2*k-1)=point(k)+0.2;
    points(:,2*k)=point;
    points(k,2*k)=point(k)-0.2;
  endfor
  pp=potential(point);
  for k=1:size(points,2)
     weights(k)=exp(-beta/2*(potential(points(:,k))-pp));
  endfor
  rate=sum(weights);
  weights=weights/rate;
endfunction
