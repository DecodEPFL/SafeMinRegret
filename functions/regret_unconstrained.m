function [Phi_x, Phi_u, objective] = regret_unconstrained(sys, sls, opt, Phi_benchmark)
%REGRET_UNCONSTRAINED computes the regret-optimal unconstrained linear
%control policy with respect to the given benchmark

    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    % Maximum eigenvalue to be minimized
    lambda = sdpvar(1, 1, 'full');
    
    % Cost incurred by the benchmark controller
    J_benchmark = Phi_benchmark.u'*kron(eye(opt.T), opt.R)*Phi_benchmark.u + Phi_benchmark.x'*kron(eye(opt.T), opt.Q)*Phi_benchmark.x;
    
    % Define objective function
    objective = lambda;
   
    constraints = [];
    % Add achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Add causal sparsities on the closed loop responses
    for i = 0:opt.T-2
        for j = i+1:opt.T-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.n):((i+1)*sys.n), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.n, sys.n)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.m, sys.n)];
        end
    end
    % Add constraints derived from Schur complement
    weights = blkdiag(sqrtm(kron(eye(opt.T), opt.Q)), sqrtm(kron(eye(opt.T), opt.R)));
    P = [eye((sys.n+sys.m)*opt.T) weights*[Phi_x; Phi_u]; [Phi_x; Phi_u]'*weights' lambda*eye(sys.n*opt.T) + J_benchmark];
    constraints = [constraints, P >= 0];
    constraints = [constraints, lambda >= 0];
    
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