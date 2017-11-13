%% Basics
help sin % help function
doc sin % matlab documentation

v = [1 2]    % define a row vector
v_T = [1 2]' % define a column vector using apostrophe

a = ones(3,4)   % define 3x4 ones matrix
a = eye(4)      % identity matrix
a = zeros(3, 4) % define 3x4 zeros matrix

a(2,1) = 1 % change an element

a(:,1) % ':' means all elements -> 1st column
a(1,:) % 1st row

a = [1 1; 1 0] % manually define a 2x2 matrix
b = [1 0; 2 1]
c = [2 1 0; 0 1 2] % a 2x3 matrix
d = [1 0 1 2; 2 1 3 1; 0 1 0 2] % a 3x4 matrix

a*b   % matrix multiplication
b*a   % and from the other side
a*c   % what will the dimensions of this matrix be?
c*d   % and this?
d*c   % and this?
a.*b  % element multiplication
a^2   % same as a*a
a.^2  % same as a.*a
a+b   % add matrices
b'    % transpose

size(a) % returns a vector of the size in each dimension

rand(3,4)   % random matrix, size 3x4. each element is U[0,1]
randn(3,4)  % random matrix, size 3x4. each element is N[0,1]

%% image show
figure;         % open a new figure
a = rand(3,4)*5 % define a new random matrix, and multipy by 5 (each element is U[0,5])
image(a);       % show the matrix a, not scaled.
colorbar

%% scaled image
clf;        % clear figure
imagesc(a); % show the matrix a, scaled.
colorbar
%%
colormap(gray)

%% matrix diag
a = [2 1; 1 0]
diag(a)         % diag(matrix) returns the diagonal of a matrix
diag([9, 7, 2]) % diag(vector) returns a matrix with vector on its diagonal
a = rand(4,4)
b = a-diag(diag(a)) % removes the diagonal from a matrix

%% plot
x = -5:0.1:5;    % define a vector with values from -5 to 5, in steps of 0.1
y = x.*sin(x); 
plot(x,y);       % plot
plot(x,y,'ro-'); % styling plots - the 1st letter is color, 2nd is marker
                 % and 3rd is the line type - for more options see
                 % "help plot"

hold on;            % to keep the 1st curve
plot(x, y+1, 'k*'); % plot black * with no line

xlabel('label for x axis','FontSize',14) % add axis labels
ylabel('label for y axis','FontSize',14)
title ('Incredible','FontSize',16);

%% reshape function
a = 1:20             % define vector from 1 to 20 in steps of 1 (default)
b = reshape(a, 5, 4) % change the shape of 'a' to a 5x4 matrix

A = magic(6);
B = reshape(A,[],3) % Reshape a 6-by-6 magic square matrix into a matrix that has only 3 columns. 
%% vectorization

% compute logarithm of 0.01,0.02,...10.01
% first method: using for loop
x = .01;
for k = 1:1001
 y(k) = log10(x);
 x = x + .01;
end
%
% vectorized version
x = .01:.01:10
y = log10(x)