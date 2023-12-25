function [position,v] = new_position(x_gbest,x_pbest,position,c1,c2,v_max,var,np,it,max_x,min_x,v,w1);
if it == 1
    x_pbest = position;
    for i = 1:np
        for j = 1:var
            v(i,j) = 0;
        end
    end
end
for i = 1:np
    v(i,:) = w1*v(i,:) + c1*rand*(x_pbest(i,:)-position(i,:)) + c2*rand*(x_gbest-position(i,:));
    for j = 1:var
        if v(i,j)>v_max(1,j);
            v(i,j) = v_max(1,j);
        elseif v(i,j)<(-v_max(1,j));
            v(i,j) = (-v_max(1,j));
        end
    end
end

position = position + v;
for i = 1:np
    for j = 1:var
        if position(i,j)>max_x(1,j);
            position(i,j) = max_x(1,j);
        elseif position(i,j)<min_x(1,j);
            position(i,j) = min_x(1,j);
        end
    end
end
return
