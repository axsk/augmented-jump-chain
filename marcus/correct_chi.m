

function val = correct_chi (val, points)
  for i=1:length(val)
    if norm(points(1:2,i)-[1;0])<0.5
          val(1,i)=1;
    elseif points(end,i)>1.6
        val(1,i)=0;
    endif
  end
endfunction
