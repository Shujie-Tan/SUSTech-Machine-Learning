load data_c3_2.mat
	
% Plot the original distribution
x = -2:0.1:14
y = gaussmf(x,[1 2])
z = gaussmf(x,[1 10])
plot(x,y)
hold on
plot(x,z)
xlabel("Guassian Mixture Model")