% size
row=112; col=92; P = 48;  % 112x92 image x 48
% timer start
tic;
% read Training image
imgbuf_Tr = zeros(row,col,P);
sum_Tr = zeros(row,col,6,'double');
total_Tr = zeros(row,col,'double');
for i = 1:1:6  % P개의 training vector를 읽음
    for j = 1:1:8
        img_path_Tr = "./Training/set_" + i + "/" + j + ".bmp";
        img_tmp = imread(img_path_Tr);
        imgbuf_Tr(:,:,8*(i-1)+j) = img_tmp(:,:,1);  % imgbuf_Tr = 모든 data 읽음 set1 부터 set6까지
        sum_Tr(:,:,i) = sum_Tr(:,:,i) + cast(img_tmp(:,:,1), 'double');       % sum_Tr = 합
        total_Tr = total_Tr + cast(img_tmp(:,:,1), 'double');
    end
end
average_Tr = sum_Tr / 6;
total_avg_Tr = total_Tr / 48;
% normalize Training Image
normalized_Tr = zeros(row,col,P);
for i = 1:1:P
    normalized_Tr(:,:,i) = imgbuf_Tr(:,:,i) - total_avg_Tr;
end
% 클래스내 분산
within_normalized_Tr = zeros(row,col,P);   %편차 112x92x48
for i = 1:1:6
    for j = 1:1:8       % 클래스 내부 편차 계산
        within_normalized_Tr(:,:,8*(i-1)+j) = imgbuf_Tr(:,:,8*(i-1)+j) - average_Tr(:,:,i);
    end
end
% 클래스간 거리
between_normalized_Tr = zeros(row,col,6); % 112x92x6
for i = 1:1:6
    between_normalized_Tr(:,:,i) = abs(average_Tr(:,:,i) - total_avg_Tr);
    %for j = 1:1:6
    %    between_normalized_Tr(:,:,i) = between_normalized_Tr(:,:,i) + abs(average_Tr(:,:,i) - average_Tr(:,:,j));
    %end
    %between_normalized_Tr = between_normalized_Tr/6;
end
% 2D -> 1D flatten
np_Tr = zeros(row*col,P);
within_np_Tr = zeros(row*col,P); % 10304x48 size
between_np_Tr = zeros(row*col,6); % 10304x6 size
for i = 1:1:P
    within_np_Tr(:,i) = reshape(within_normalized_Tr(:,:,i), [row*col,1]);
    np_Tr(:,i) = reshape(normalized_Tr(:,:,i), [row*col,1]);
end
for i = 1:1:6
    between_np_Tr(:,i) = reshape(between_normalized_Tr(:,:,i), [row*col,1]);
end
% 공분산 행렬
within_cov_Tr = within_np_Tr * within_np_Tr.';  % D_intra
between_cov_Tr = between_np_Tr * between_np_Tr.';   % D_inter
% eigenvalue & vector -> v
[e_vector, e_value] = eigs((pinv(within_cov_Tr)*between_cov_Tr), 5);
% v에 projection된 거리 & 분산 -> v_t * D
pj_Tr = zeros(5,P);
for i = 1:1:P   % 3x10304 * 10304x48 = 3x48
    pj_Tr(:,i) = e_vector.' * np_Tr(:,i);
end
% Read Test Image
imgbuf_Te = zeros(row,col,12);
for i = 1:1:12
    img_path_Te = "./Test/" + i + ".bmp";
    img_tmp = imread(img_path_Te);
    imgbuf_Te(:,:,i) = img_tmp(:,:,1);
end
% normalize Test Image
normalized_Te = zeros(row,col,12);
for i = 1:1:12
    normalized_Te(:,:,i) = imgbuf_Te(:,:,i) - total_avg_Tr;
end
% flatten Test Image
np_Te = zeros(row*col,12);
for i = 1:1:12
    np_Te(:,i) = reshape(normalized_Te(:,:,i), [row*col,1]);
end
% mapping
pj_Te = zeros(5,12);
for i = 1:1:12
    pj_Te(:,i) = e_vector.' * np_Te(:,i);  % 3x12
end
% Euclidean Distance
ed = zeros(12, 48);
for i = 1:1:12
    for j = 1:1:48
        dist = pj_Te(:,i) - pj_Tr(:,j);
        ed(i, j) = sum(dist.*dist, 'all');  % element-wise square
    end
end
% classification
[result, index] = min(ed, [], 2);
% 정답은 2,2,1,1,3,3,4,4,5,5,6,6이 나와야 함
% print result
answer = fix((index - 0.1)/8) + 1;
% stop timer
toc;

