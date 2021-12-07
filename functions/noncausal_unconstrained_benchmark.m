function [Phi_x, Phi_u, obj_h2, obj_hinf] = noncausal_unconstrained_benchmark(sys, sls, opt)
%NONCAUSAL_UNCONSTRAINED_BENCHMARK computes the optimal dynamic sequence of
%control actions

    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    % Define objective function
    objective = norm([sqrtm(kron(eye(opt.T), opt.Q))*Phi_x; sqrtm(kron(eye(opt.T), opt.R))*Phi_u], 'fro');
    
    constraints = [];
    % Add achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    obj_h2 = value(objective); % H2-optimal cost incurred by the clairvoyant controller
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    %%%%% Solve another instance of the problem to compute the Hinf-optimal 
    %%%%% cost incurred by the clairvoyant controller 
    
    % Define objective function
    objective = norm([sqrtm(kron(eye(opt.T), opt.Q))*Phi_x; sqrtm(kron(eye(opt.T), opt.R))*Phi_u], 2);
    
    % Solve the optimization problem
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    obj_hinf = value(objective); % Hinf-optimal cost incurred by the clairvoyant controller
end