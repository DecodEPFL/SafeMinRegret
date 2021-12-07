function [x, u, ta_cum_costs] = evaluate_policy(sys, opt, Phi, w)
%EVALUATE_POLICY computes the input-state trajectory generated applying the
%policy Phi in response to the disturbance realization w, and the incurred 
%time-averaged cumulative costs
    
    % Compute the input-state trajectory associated with the disturbance w 
    x = Phi.x * w; 
    u = Phi.u * w; 
    % Reshape the input-state trajectory for simplicity
    x = reshape(x, [sys.n, opt.T]);
    u = reshape(u, [sys.m, opt.T]);
    
    % Compute the time-averaged cumulative stage costs along the control horizon
    ta_cum_costs = zeros(1, opt.T);
    ta_cum_costs(1) = norm(x(:, 1), 2)^2 + norm(u(:, 1), 2)^2;
    for i = 2:opt.T
        ta_cum_costs(i) = ((i-1)*ta_cum_costs(i-1) + norm(x(:, i), 2)^2 + norm(u(:, i), 2)^2)/i;
    end
    
end

