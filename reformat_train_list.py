import csv

train_list = csv.reader(open('Sony_train_list.txt'), delimiter=" ")
new_list = []

for line in train_list:
    l1 = line[1].split('.')
    line[1] = '.' + l1[1] + '.png'
    new_list.append(line)

with open('train_list_pngs.txt', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(new_list)