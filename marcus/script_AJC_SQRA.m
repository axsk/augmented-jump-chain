%% SETTINGS
num_points=300;
num_cycles=50;

%% training points in space and time
train=rand(2,num_points)*4-2;
train=[train;rand(1,num_points)*2];

%% Initialize chi-values randomly 
val_chi=rand(1,num_points); 

%% Correction of chi according to cores A and B
val_chi=correct_chi(val_chi, train);

for i=1:num_cycles
  %% Interpolation of chi-values in space and time
  %% at training points
  rbf=rbfcreate(train, val_chi, 'RBFFunction', 'multiquadratic', 'RBFConstant', 2);
  
  %% set new chi value at each point
  for j=1:num_points
    %% search for all spatial neighbors and jump probabilities
    [train_neighbors, weights, rate]=create_neighbors(train(:,j));
    
    %% add time-intervall
    [train_neighbors, weights]=correct_time(train_neighbors, weights, rate);
    
    %% interpolate chi-values at neighbors
    val_chi_neighbors=abs(rbfinterp(train_neighbors, rbf));
    
    %% correct chi_values according to cores A and B
    val_chi_neighbors=correct_chi(val_chi_neighbors, train_neighbors);    
    
    %% compute matrix multiplication by taking the mean
    val_chi_new(j)=sum(val_chi_neighbors.*weights);
  endfor
  plot(val_chi,val_chi_new,'*');
  pause(0.5);
  
  cc=corrcoef(val_chi, val_chi_new);
  if (cc(1,2)>0.999)
    break;
  endif

  if (mod(i,5)==0)
    %% new training points in space and time
    train=rand(2,num_points)*4-2;
    train=[train;rand(1,num_points)*2];

    %% interpolate chi-values 
    val_chi=abs(rbfinterp(train, rbf)); 

    %% Correction of chi according to cores A and B
    val_chi=correct_chi(val_chi, train);
  else
    val_chi=val_chi_new;
  endif 
endfor

figure(1)
for i=0:0.2:2
  val_interp=rbfinterp([train(1:2,:);i*ones(1,size(train,2))], rbf);
  illustrate(train(1:2,:),abs(val_interp),50);
  pause(1)
end

figure(2)
for i=1:length(val_chi_new)
  pot_val(i)=potential(train(1:2,i));
end
illustrate(train(1:2,:),pot_val,50);

