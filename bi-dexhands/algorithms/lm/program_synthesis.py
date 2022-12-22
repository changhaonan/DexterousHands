import copy

def parse_command_file(command_file):
    command_list = []
    with open(command_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):  # comment
                continue
            line_list = line.upper().split(" ")
            # check line list and convert number to float
            for i, word in enumerate(line_list):
                if word.startswith("("):
                    number_f = [float(number) for number in word[1:-1].split(",")] 
                    line_list[i] = number_f

            command_list.append(line_list)
    return command_list

def recursive_synthesis(sketch_candidate, vocubulary):
    flag = False

    # find ?? in sketch
    line_idx = 0 
    word_idx = 0
    candidate_idx = 0
    for i, sketch in enumerate(sketch_candidate):
        for j, line in enumerate(sketch):
            if '??' in line:
                flag = True
                line_idx = j
                word_idx = line.index('??')
                break
        if flag:
            break

    if flag:
        # replace ?? with vocabulary   
        for word in vocubulary:
            sketch_candidate.append(copy.deepcopy(sketch_candidate[candidate_idx]))
            sketch_candidate[-1][line_idx][word_idx] = word
        del sketch_candidate[candidate_idx]
        recursive_synthesis(sketch_candidate, vocubulary)
    else:   
        return 


if __name__ == "__main__":
    sketch_file = "/home/robot-learning/Projects/DexterousHands/bi-dexhands/test/grasp_place.sketch"
    sketch = parse_command_file(sketch_file)
    vocubulary = ['GRASP', 'PLACE', 'RELEASE']
    sketch_candidate = [sketch]
    recursive_synthesis(sketch_candidate, vocubulary)