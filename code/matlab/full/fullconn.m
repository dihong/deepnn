% X is Mx1 matrix consisting of one training point.
% W is QxM matrix where M is input dim, Q is output dim.
% b is Qx1 vector for bias.

%The gEgY is 1xQ vector represents the gradient of E w.r.t. output of this
% full layer. 

%reference: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

function [Y, gEgX, gEgW, gEgb] = fullconn(X,W,b,gEgY,activation)
if strcmp(activation,'tanh')
    Y = tanh(bsxfun(@plus,W*X,b));  %Qx1
    gEgb = gEgY.*(1-Y.^2); %Qx1
    gEgW = gEgb*X(:)'; %QxM
    gEgX = gEgb'*W;
elseif strcmp(activation,'relu')
    
else
    fprintf('unknown activation function: %s\n',activation);
    return;
end
end