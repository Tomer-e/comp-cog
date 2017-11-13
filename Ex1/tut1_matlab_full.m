%% example of function
A = randn(6)
A_ZERO_DIAG = remove_diag(A)
%% Now a small hopfield
A = [1, 1, 1, 1; 1, 1, -1, -1; 1, -1, 1, -1]' % each column is a memory pattern
N = size(A,1)
P = size(A,2)
%% Compute the connections matrix, the hard way:
J = zeros(N,N);
for i=1:N
    for j=1:N
            for mu=1:P
                J(i,j) = J(i,j) + 1/N * A(i,mu) * A(j,mu);
            end
    end
end
J = remove_diag(J);
% zerp diagonal (no self connection)

%% Compute the connections matrix, the easy way:
J2 = 1/N*(A*A');
J2 = remove_diag(J2);
%% Did we get the same result?
all(all(J2==J))

%% Check if memory pattern 1 is stable
% S is the state of the network

% initialize S to the 1st memory pattern
S = A(:,1);

% iterate over all neurons
for i = 1:N
    % asynchronously update each neuron
    S(i) = sign(J(i,:)*S);
    if S(i)==0 % choose what to do in case of zero
        S(i)= 1;
    end
end

% is S still the 1st memory pattern?
all(S == A(:,1))

%% Check stability for all the memory patterns

% run over all memory patterns
for mu = 1:P
    % initializing
    S = A(:,mu);
    
    % iterate over all neurons (asynchronous update)
    for i = 1:N
        S(i) = sign(J(i,:)*S)
        if S(i)==0
            S(i)= 1;
        end
    end
    
    % is S still memory pattern mu?
    all(S==A(:,mu))
end


%% Larger hopfield network
N = 30;
P = 4;

%% Random memory patterns and connections matrix 
% Generate unbiased and uncorrelated memory patterns
A = (rand(N,P)<0.5)*2-1; 

J = 1/N*(A*A');
J = remove_diag(J);
%% Initialize the network from a random state, and run until it converges
S = (rand(N,1)<0.5)*2-1;

con = 0;
while ~con
    S_old = S ;
    
    % iterate over all neurons
    for i = 1:N
        S(i) = sign(J(i,:)*S);
        if S(i)==0
            S(i)= -1;
        end
    end
    
    % convergence test
    if all(S_old == S)
        con = 1;
    end
end

%% Check to which memory pattern the network was converged to
result = 0;

for mu = 1:P
    
    if all(S == A(:,mu))
        result = mu;
    elseif all(S == -A(:,mu))
        result = -mu;
    end
end
disp(['Memory pattern number ' num2str(result)]);

%% Add probabilistically 10% noise
N = 10;
P = 2;
A = (rand(N,P)<0.5)*2-1
noise_mat = (2*(rand(size(A))>0.1)-1);
B = A .* noise_mat;

%% Add exactly 10% noise 
% Random memory patterns
N = 10;
P = 2;
A = (rand(N,P)<0.5)*2-1

% number of cells we want to change
num = round(N*0.1);

% vector with -1 (n times) and 1 (length(A)-n times)
noise_vec = [-ones(1, num), ones(1, length(A)-num)]';

noise_mat = zeros(N,P);
for mu = 1:P
    % shuffle the vector randomly for each memory pattern
    noise_vec = noise_vec(randperm(length(noise_vec)));
    noise_mat(:,mu) = noise_vec;
end

B = A.*noise_mat;

