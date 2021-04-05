# tf - term frequency
# idf - inverse document frequency
# they convert sentences into vectors
import math
from os import listdir
from os.path import isfile, join
import sys
import re

def get_termfreqs(Openedfile):
    Openedfile.seek(0)
    map1 = {}
    for i in Openedfile:
        k = i.split(" ")
        term = k[3]
        term = term.strip()
        if term in map1:
            map1[term]=map1.get(term)+1
        else:
            map1[term]= 1
    sum1 = 0
    for i in map1:
        sum1 += map1[i]
    # print(sum1)
    for i in map1:
        map1[i] = map1[i]/sum1
    # print(ans)
    return map1

def find_sensors_with_term(Openedfile, term):
    set1 = set()
    for i in Openedfile:
        k = i.split(" ")
        sensor_id = k[1]
        sensor_id = sensor_id.strip()
        elem = k[3]
        elem = elem.strip()
        if(term == elem):
            set1.add(sensor_id)
    return len(set1)

def idf(no_of_sensors, sensors_with_term):
    if(sensors_with_term == 0): return 0.0
    return math.log(no_of_sensors/sensors_with_term)

def tfidf(tf, idf1, maxtf):
    return (0.5+ 0.5*tf/maxtf)*idf1

def tfidf2(tf, idf2, maxtf):
    return (0.5+ 0.5*tf/maxtf)*idf2

def find_maxidf(idfs):
    return max(idfs)

def find_all_levels(Openedfile):
    set1 = set()
    for i in Openedfile:
        k = i.split(" ")
        level = k[3]
        level = level.strip()
        set1.add(level)
    # print(set1)
    return set1

def find_no_of_sensors(Openedfile):
    set1 = set()
    for i in Openedfile:
        k = i.split(" ")
        sensor_id = k[1]
        sensor_id = sensor_id.strip()
        if(not sensor_id in set1):
            set1.add(sensor_id)
    return len(set1)

def find_all_terms_inDir(directory):
    count = 0
    files = get_wrd_files(directory)
    set1 = set() 
    for file in files:
        if file.endswith('.wrd'):
            Openedfile = open(directory+file, "r")
            # print(Openedfile)
            set_temp = find_all_levels(Openedfile)
            # print(set_temp)
            set1 = set1.union(set_temp)
            # print(set1)
    # print(set1)
    return set1

def get_maxtf(i, Openedfile):
    map1 = {}
    totalcount = 0
    for line in Openedfile:
        k = line.split(" ")
        sensor_id = k[1]
        sensor_id = sensor_id.strip()
        value = k[3]
        value = value.strip()
        if(sensor_id == i and value in map1):
            # print(sensor_id)
            map1[value] += 1
            totalcount+=1
        elif (sensor_id == i and value not in map1):
            map1[value] = 1
            totalcount+=1
    # print(map1.values())
    return max(map1.values())/totalcount


def fill_idf_vals(Openedfile, levels):
    idf_vals = []
    for term in levels:
        Openedfile.seek(0)
        no_of_sensors = find_no_of_sensors(Openedfile)
        Openedfile.seek(0)
        sensors_with_term = find_sensors_with_term(Openedfile, term)
        idf_vals.append(idf(no_of_sensors, sensors_with_term))

def find_no_of_files(directory):
    count = 0
    files = get_wrd_files(directory)
    for file in files:
        if file.endswith('.wrd'):
            count+=1
    return count

def find_files_with_term(directory, elem):
    count = 0
    files = get_wrd_files(directory)
    for file in files:
        if file.endswith('.wrd'):
            Openedfile = open(directory+file, "r")
            # print(Openedfile)
            for i in Openedfile:
                # print(i)
                k = i.split(" ")
                # print(k)
                term = k[3]
                term = term.strip()
                if(term == elem):
                    count+=1
                    break
    return count


def fill_idf2_vals(Openedfile, levels):
    idf2_vals = []
    for term in levels:
        Openedfile.seek(0)
        no_of_files = find_no_of_files(directory)
        Openedfile.seek(0)
        files_with_term = find_files_with_term(directory, term)
        idf2_vals.append(idf(no_of_files, files_with_term))


# def write_output2():



def write_output(no_of_sensors, Openedfile,levels, task2_output, directory):
    Openedfile.seek(0)
    tf_map = get_termfreqs(Openedfile)
    tf_vec = []
    idf1_vec = []
    idf2_vec = []
    for j in levels:
        Openedfile.seek(0)
        temp_tf = tf_map.get(j,0)
        # print(temp_tf)
        tf_vec.append(temp_tf)
        Openedfile.seek(0)
        temp_idf = idf(no_of_sensors, find_sensors_with_term(Openedfile, j))
        temp_maxtf1 = float(max(tf_map, key=tf_map.get))
        # print(type(temp_tf))
        temp_tfidf1 = tfidf(temp_tf,temp_idf, temp_maxtf1)
        idf1_vec.append(temp_tfidf1)
        Openedfile.seek(0)
        # temp_maxtf = get_maxtf(str(i), Openedfile)
        temp_idf2 = idf(find_no_of_files(directory), find_files_with_term(directory, j))
        temp_maxtf2 = float(max(tf_map, key=tf_map.get))
        temp_tfidf2 = tfidf2(temp_tf, temp_idf2, temp_maxtf2)
        idf2_vec.append(temp_tfidf2)
    a = str(", ".join(repr(e) for e in tf_vec))
    b = str(", ".join(repr(e) for e in idf1_vec))
    c = str(", ".join(repr(e) for e in idf2_vec))
    task2_output.write("< "+a +" >"+"< "+b+" >"+"< "+c +" >")
    task2_output.write("\n")
    Openedfile.close()
 

# Openedfile = open("output/1.wrd", "r")

# write vectors such that it also includes zero counts
def write_all_vectors(Openedfile, task2_output, directory):
    # levels = find_all_levels(Openedfile)
    levels = find_all_terms_inDir(directory)
    # print("type of variable levels is:")
    # print(type(levels))
    # print(levels)
    levels = sorted(levels)
    Openedfile.seek(0)
    # fill_idf_vals(Openedfile, levels)
    Openedfile.seek(0)
    no_of_sensors = find_no_of_sensors(Openedfile)
    Openedfile.seek(0)
    write_output(no_of_sensors, Openedfile, levels, task2_output, directory)

def get_wrd_files(directory):
    only_csv_files = [f for f in listdir(directory) if isfile(
        join(directory, f)) if f.endswith('.wrd')]
    return only_csv_files



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def gesture_vectors(directory):
    files = get_wrd_files(directory)
    sort_nicely(files)
    # print(files)
    open(directory+'vectors.txt', 'w').close()
    task2_output = open(directory+"vectors.txt", "a")
    for file in files:
        # print("file is")
        # print(file)
        Openedfile = open(directory+file, "r")
        ans = write_all_vectors(Openedfile, task2_output, directory)
    task2_output.close()

directory = "../codes/output/"
# k = gesture_vectors(directory)

if __name__ == "__main__":
    # Openedfile = open(directory+file, "r")
    # levels = find_all_levels(Openedfile)
    # print(levels)
    directory = sys.argv[1]
    print("wait for a few seconds")
    gesture_vectors(directory)
    print("done! check the vector file")

# python task2.py ../codes/output/
