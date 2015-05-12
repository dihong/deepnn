function test
addpath('../output/softmax');
X = rand(1,4);
labels = [1 2 1 2 2];
nc = 2;
W = rand(nc,size(X,2));
b = zeros(nc,1);
gEgY = zeros(nc,1);
activation = 'tanh';
for i = 1:50
    gEgW = 0;
    gEgb = 0;
    Y = [];
    gEgX = [];
    for n = 1:size(X,1) %for each training sample.
        [y, x, w, bb] = fullconn(X(n,:)',W,b,gEgY,activation);
        Y = [Y; y'];
        gEgX = [gEgX ; x];
        gEgW = gEgW + w;
        gEgb = gEgb + bb;
    end
    gEgW = gEgW / size(X,1);
    gEgb = gEgb / size(X,1);
    [E(i), gEgY] = softmax(Y, labels);
%     b = b - 0.5*gEgb;
%     X = X - 0.5*gEgX;
%     W = W - gEgW;
end
plot(E(2:end))
xlabel('cost')
ylabel('iteration')
end