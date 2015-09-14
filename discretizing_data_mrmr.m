Comb_matrix = Slope_comb_matrix;
matrix_mean = mean(Comb_matrix,2);
matrix_std = std(Comb_matrix,0,2);

mul = 1;
disc_Comb_matrix = ones(size(Comb_matrix));
disc_Comb_matrix(Comb_matrix < 0) = -1;
% disc_Comb_matrix = zeros(size(Comb_matrix));
% for i = 1:size(Comb_matrix,1)
     %disc_Comb_matrix(i,(Comb_matrix(i,:) <= matrix_mean(i) - mul*matrix_std(i))) = -2;
     %disc_Comb_matrix(i,(Comb_matrix(i,:) > matrix_mean(i) + mul*matrix_std(i))) = 2;
% end
      
% k = 100;    
% figure; plot(sort(Comb_matrix(k,:)))
% hold on; plot(sort(disc_Comb_matrix(k,:)),'r')
% line([0 320],[matrix_mean(k) - mul*matrix_std(k) matrix_mean(k) - mul*matrix_std(k)])
% line([0 320],[matrix_mean(k) + mul*matrix_std(k) matrix_mean(k) + mul*matrix_std(k)])

%disc_Comb_matrix = Comb_matrix;
topNfeatures = 1; %size(disc_Comb_matrix,1);
[rank_combo,mrmrMI] = mrmr_mid_d(disc_Comb_matrix',class_labels',topNfeatures);
Ranked_Channel_comb = Channel_comb_matrix(rank_combo,1)'
figure; plot(Ranked_Channel_comb,-mrmrMI);hold on; plot(Ranked_Channel_comb,-mrmrMI,'x');