import csv

class CSV_Parser:        
    def get_keys_and_data(self, csvname):
        with open(csvname, 'r') as csvfile:
            textfile = open(csvname, 'r')
            lines = textfile.read().split('\n')
            keys = lines[0].split(',')
            data = []
            for i in range(len(lines)-1):
                data.append(lines[i+1].split(','))
            return keys, data