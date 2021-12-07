function [Phi_x, Phi_u, objective] = noncausal_constrained_benchmark(sys, sls, opt, flag)
%NONCAUSAL_CONSTRAINED_BENCHMARK computes the H2- or Hinf-optimal noncausal
%constrained linear control policy

    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    % Define objective function
    if strcmp(flag, 'H2')
        objective = norm([sqrtm(kron(eye(opt.T), opt.Q))*Phi_x; sqrtm(kron(eye(opt.T), opt.R))*Phi_u], 'fro');
    elseif strcmp(flag, 'Hinf')
        objective = norm([sqrtm(kron(eye(opt.T), opt.Q))*Phi_x; sqrtm(kron(eye(opt.T), opt.R))*Phi_u], 2);
    else
        error('Something went wrong...');
    end
   
    constraints = [];
    % Add achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I]; 
    % Add polytopic safety constraints
    z = sdpvar(size(sls.Hw, 1), size(sls.H, 1), 'full'); % Dual variables
    for i = 1:size(sls.H, 1)
        constraints = [constraints, z(:, i)'*sls.hw <= sls.h(i)];
        constraints = [constraints, z(:, i) >= 0];
    end
    constraints = [constraints, sls.H*[Phi_u; Phi_x] == z'*sls.Hw];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    objective = value(objective);
end