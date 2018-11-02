import sys
dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
for n_data, data in enumerate(dataset):
    outputs = []
    relations = {}
    entities = []
    lines = data.split('\n')
    for n_line, line in enumerate(lines):
        elements = line.strip().split(" ")
        if len(elements) == 6 and elements[0] == "token":
            entities.append(elements)
        elif len(elements) == 5 and elements[0] == "rel":
            relation = elements[4]
            if int(elements[3]) == 1:
                head = int(elements[1])
                tail = int(elements[2])
            elif int(elements[3]) == -1:
                head = int(elements[2])
                tail = int(elements[1])
            if tail not in relations:
                relations[tail] = []
            relations[tail].append(str(head) + "$$" + relation)
    for i in range(0,len(entities)):
        entity = entities[i]
        if entity[5] == "o":
            print str(i+1) + "\t" + entity[1] + "\t" + entity[1] + "\t" + entity[2] + "\t" + entity[2] + "\t_\t" + str(i+1) + "\t" + "O-DELETE"
            if i in relations:
                print "ERROR"
        elif entity[5][0] in ["b","m"]:
            print str(i+1) + "\t" + entity[1] + "\t" + entity[1] + "\t" + entity[2] + "\t" + entity[2] + "\t_\t" + str(i+1) + "\t" + "GS"
            if i in relations:
                print "ERROR"
        elif entity[5][0] in ["s","e"]:
            print str(i+1) + "\t" + entity[1] + "\t" + entity[1] + "\t" + entity[2] + "\t" + entity[2] + "\t_\t" + str(i+1) + "\t" + "GN(" + entity[5][2:] + ")"
            if i in relations:
                for el in relations[i]:
                    tokens = el.strip().split("$$")
                    print str(i+1) + "\t" + entity[1] + "\t" + entity[1] + "\t" + entity[2] + "\t" + entity[2] + "\t_\t" + str(int(tokens[0]) + 1) + "\t" + tokens[1]
    print ""


