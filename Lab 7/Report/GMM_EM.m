function [mean,cov,coef,p_for_GMM] = GMM_EM(X,K)
%input:
%   X:N*D matrix ,N data points in a D dimension space.
%   K:The number of Gaussian you expected in the model.
%output:
%   mean:K*D matric,means of the Gaussian in the model
%   cov:D*D*K matric, covariance of Gaussian in the model
%   coef:K dimension vector,weight of each Gaussian in the model
%   p_for_GMM:The probability of data points belong to each Gaussian in the
%   model.
%   write by JinYiKang 2017/9/13.
[N,D] = size(X);

%initilization parameters
mean = eye(K,D);
cov = zeros(D,D,K);
for i=1:K
    cov(:,:,i) = eye(D,D);
end
coef = ones(1,K)/K;

p = 0;
thresh = 0.1;
while 1
    gv = zeros(N,K); % The vaule of K Gaussian on data points.
    for l = 1:K
        gv(:,l) = gaussian(X,mean(l,:),cov(:,:,l));
    end
    denominator = gv*coef';
    
    % check for convergence of the log likelihood.
    if abs(sum(log(denominator)) - p) > thresh
        p = sum(log(denominator));
    else
        break;
    end
    
    for j=1:K
%E-step:evaluate the responsibilities using the current parameter values.  
        numerator = coef(j)*gv(:,j);
        re = numerator./denominator; % evaluate the responsibilities
        Nk = sum(re);
%M-step:Re-estimate the parameters using the current responsibilities.    
        mean(j,:) = 1/Nk*re'*X;
        cov(:,:,j) = 1/Nk*(X-repmat(mean(j,:),N,1))'*diag(re)*(X-repmat(mean(j,:),N,1));
        coef(j) = Nk/N;
    end
end

p_for_GMM = gv.*repmat(coef,N,1)./repmat(denominator,1,K);
    function v = gaussian(X,mean,cov)
        xtmp = X - repmat(mean,N,1);
        b = diag(xtmp*inv(cov)*xtmp');
        a = (2*pi)^(D/2)*sqrt(det(cov));
        v = 1/a*exp(-0.5*b);
    end
end