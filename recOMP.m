function [C R I] = recOMP(A,x,K)
%------------------------------------------------------------------------------------
% Orthogonal Matching Pursuit
%
% Usage:
%       [C R I] = recOMP(A,x,K)
%
% Input:
%   X:      d x n matrix of training samples
%           n --- the number of training samples 
%           d --- dimensionality of samples%
%   x:      d x 1 test sample
%   K:      sparsity level
%
% Output:
%   C:      coefficient vector 
%   R:      residual
%   I:      indices of selected atoms
% 
%------------------------------------------------------------------------------------
% Author & Affiliation:
% Minshan Cui, Department of Electrical and Computer Enginnering, University of Houston.
% Hyperspectral Image Analysis Lab: http://hyperspectral.ee.uh.edu/
%------------------------------------------------------------------------------------

[d n] = size(A);

% normalization
A = l2norm(A);
x = l2norm(x);

% initialization
res = x;
I = [];
A_I = [];
C = zeros(n,1);

% iteration
for i = 1 : K
    [~,ind_sel] = max(abs(A'*res));
    atom_sel = A(:,ind_sel);
    A_I = [A_I atom_sel];
    
    I = [I ind_sel];
    C(I) = pinv(A_I) * x;
    
    res = x - A*C;
end

R = norm(x-A*C,'fro');


