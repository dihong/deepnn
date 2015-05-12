function test
addpath('../../../Data/MNIST');
X = images(1:10,1)';
alpha = 0.5;
for i = 1:50
    [Y(i), gEgX] = softmax(X, labels+1);
    X = bsxfun(@minus,X,alpha*gEgX');
end
plot(Y)
xlabel('iteration')
ylabel('error')
title('softmax')
end