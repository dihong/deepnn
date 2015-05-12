function cnn
addpath('../../../Data/MNIST');
addpath('../softmax');
images = loadMNISTImages('Data/MNIST/train-images.idx3-ubyte');
labels = loadMNISTLabels('Data/MNIST/train-labels.idx1-ubyte');
X = images(1:10,1:100)';
alpha = 1e-2;
for i = 1:50
    [Y(i), gEgX] = unit(X, labels+1);
    X = bsxfun(@minus,X,alpha*gEgX');
end
plot(Y)
xlabel('iteration')
ylabel('error')
title('softmax')
end

% X is NxD matrix consisting of N training samples. D is the depth of the
% network. Y is the negative log likelihood representing the error.
function [Y, gEgX] = unit(X, label)
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
end