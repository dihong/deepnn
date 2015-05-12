% X is 1xD matrix consisting of 1 training sample. D is the depth of the
% network. Y is the negative log likelihood representing the error.
function [Y, gEgX] = softmax(X, label)
Y = 0;
gEgX = 0;
eX = exp(X);
for n = 1:size(X,1)
    Y = Y - log( eX(n,label(n)) / sum(eX(n,:)));
    one_of_k_label = zeros(size(X,2),1);
    one_of_k_label(label(n)) = 1;
    gEgX = gEgX + eX(n,:)' / sum(eX(n,:)) - one_of_k_label; %according to Facebook-AI-Intuition.
end
Y = Y / size(X,1);
gEgX = gEgX / size(X,1);
end