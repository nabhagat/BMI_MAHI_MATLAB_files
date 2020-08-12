function lab_pre = srclassifier(X1,X2,Y1,rec,arg)
%------------------------------------------------------------------------------------
% Sparse Representation Classifier
%
% Usage:
%       lab_pre = srclassifier(X1,X2,Y1,rec,arg)
%
% Input:
%    X1:     d x n1 matrix of training samples
%            n1 --- the number of training samples 
%            d --- dimensionality of samples
%    X2:     d x n2 matrix of training samples
%            n1 --- the number of training samples 
%            d --- dimensionality of samples
%    Y1:     n1 dimensional vector of training sample labels
%    rec:    type of reconstruction method
%            'BP    --- basis pursuit
%            'BPDN' --- basis pursuit denoising
%            'OMP'  --- orthogonal matching pursuit
%            'SP'   --- subspace pursuit
%    arg:    argument used in reconstruction method 
%
% Output:
%    lab_pre: predicted labels 
% 
%------------------------------------------------------------------------------------
% Author & Affiliation:
% Minshan Cui, Department of Electrical and Computer Enginnering, University of Houston.
% Hyperspectral Image Analysis Lab: http://hyperspectral.ee.uh.edu/
%------------------------------------------------------------------------------------

% l2normalization
X1 = l2norm(X1);
X2 = l2norm(X2);

% initialization
n2 = size(X2,2);
cls_lab = unique(Y1);
cls_num = length(cls_lab);
A = [];
for i = 1 : cls_num
    ind_cls = find(Y1==i);
    A = [A, X1(:,ind_cls)];
end

% main loop
for i = 1 : n2
    b = X2(:,i);
    switch rec
        %BP 
        case 'BP' % l1 toolbox
            x0=ones(n1,1);
            x = l1eq_pd(x0, A, 1, b);
        %BPDN 
        case 'BPDN' % spgl1 toolbox
            %sigma = options.spgl1_sigma;
            sigma = arg(1);
            opts = spgSetParms('verbosity',0);
            x = spg_bpdn(A, b, sigma, opts);
        %OMP 
        case 'OMP' % sparseLab toolbox
            K = arg(1);
            x = recOMP(A, b, K);
            %x = SolveOMP(A, b, n1, K);
        %SP
        case 'SP' % SP toolbox
            K = arg(1);
            %x = SP(K, A, b, n1);
            x = CSRec_SP(K,A,b);
    end
    
    % calculate residule
    res_vec = zeros(1,cls_num);
    ctr = 0;
    for z = 1 : cls_num
        tmp_num  = length(find(Y1==cls_lab(z)));
        tmp_coef = x(ctr+1:ctr+tmp_num);
        res_vec(z) = norm(b - A(:,ctr+1:ctr+tmp_num)*tmp_coef,'fro');
        ctr = tmp_num + ctr;
    end
    [~,lab_pre(i,1)] = min(res_vec);
end


