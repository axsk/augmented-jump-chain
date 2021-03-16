

function [points_out, weights_out] = correct_time (points, weights, rate)
    num_samp=size(points,2);
    points_out=zeros(size(points,1), size(points,2)*10);
    weights_out=zeros(1, size(points,2)*10);
    X1=rand(1,num_samp*10);
    t1= (- 1/rate)*log(1-X1);
    for i=1:num_samp
      for k=1:10
          points_out(:,10*(i-1)+k)=points(:,i);
          points_out(end,10*(i-1)+k)=points_out(end,10*(i-1)+k)+t1(10*(i-1)+k);
          weights_out(10*(i-1)+k)=weights(i)/10;
      end
    end 
endfunction
