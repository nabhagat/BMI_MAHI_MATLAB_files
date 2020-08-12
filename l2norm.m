function data=l2norm(data)

num = size(data,2);
for i = 1 : num
    data(:,i) = data(:,i) / norm(data(:,i),'fro');
end