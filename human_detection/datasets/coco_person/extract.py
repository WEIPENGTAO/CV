# 取train2017_person.txt的前5000条数据，生成train2017_person_5000.txt
with open('train2017_person.txt', 'r') as f:
    lines = f.readlines()
    with open('train2017_person_5000.txt', 'w') as f2:
        for line in lines[:5000]:
            f2.write(line)

# 取val2017_person.txt的前1000条数据，生成val2017_person_1000.txt
with open('val2017_person.txt', 'r') as f:
    lines = f.readlines()
    with open('val2017_person_1000.txt', 'w') as f2:
        for line in lines[:1000]:
            f2.write(line)